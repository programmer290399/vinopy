#!/usr/bin/env python
import os
import cv2
import sys
import time
from typing import NewType
import logging as log

try:
    from openvino.inference_engine import IENetwork, IECore
except:
    log.info("OpenVINO ERROR: Please make sure that OpenVINO is installed properly")
    sys.exit(1)


class VinoInfer:

    def __init__(self,
                 model_path,
                 model_xml,
                 input_stream,
                 extention_lib_path='/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so',
                 device='CPU',
                 labels=None,
                 prob_thresh=0.5,
                 async_mode=False):

        self.model_path = model_path
        self.model_xml = model_xml
        self.device = device
        self.input_stream = input_stream
        self.extention_lib_path = extention_lib_path
        self.labels = labels
        self.prob_thresh = prob_thresh
        self.async_mode = async_mode

    def draw_inference(self):
        """
        Call this functions after creating object of VideoInfer
        class by passing all the required parameters to draw inference.
        """

        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        log.info("Creating Inference Engine...")
        ie = IECore()
        if self.extention_lib_path and 'CPU' in self.device:
            ie.add_extension(self.extention_lib_path, "CPU")
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, self.model_path))
        net = IENetwork(model=self.model_xml, weights=self.model_path)

        if "CPU" in self.device:
            supported_layers = ie.query_network(net, "CPU")
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        img_info_input_blob = None
        feed_dict = {}
        for blob_name in net.inputs:
            if len(net.inputs[blob_name].shape) == 4:
                input_blob = blob_name
            elif len(net.inputs[blob_name].shape) == 2:
                img_info_input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                   .format(len(net.inputs[blob_name].shape), blob_name))

        assert len(net.outputs) == 1, "Demo supports only single output topologies"

        out_blob = next(iter(net.outputs))
        log.info("Loading IR to the plugin...")
        exec_net = ie.load_network(network=net, num_requests=2, device_name=self.device)
        # Read and pre-process input image
        n, c, h, w = net.inputs[input_blob].shape
        if img_info_input_blob:
            feed_dict[img_info_input_blob] = [h, w, 1]

        if self.input_stream == 'cam':
            input_stream = 0
        elif self.input_stream.startswith('rtsp'):
            log.info('Using RTSP feed')
            input_stream = self.input_stream
        else:
            input_stream = self.input_stream
            assert os.path.isfile(self.input_stream), "Specified input file doesn't exist"
        if self.labels:
            with open(self.labels, 'r') as f:
                labels_map = [x.strip() for x in f]
        else:
            labels_map = None

        cap = cv2.VideoCapture(input_stream)

        cur_request_id = 0
        next_request_id = 1

        log.info("Starting inference in async mode...")
        
        render_time = 0
        ret, frame = cap.read()

        print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
        print("To switch between sync/async modes, press TAB key in the output window")

        while cap.isOpened():
            if self.async_mode:
                ret, next_frame = cap.read()
            else:
                ret, frame = cap.read()
            if not ret:
                break
            initial_w = cap.get(3)
            initial_h = cap.get(4)
            # Main sync point:
            # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            # in the regular mode we start the CURRENT request and immediately wait for it's completion
            inf_start = time.time()
            if self.async_mode:
                in_frame = cv2.resize(next_frame, (w, h))
                in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                in_frame = in_frame.reshape((n, c, h, w))
                feed_dict[input_blob] = in_frame
                exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
            else:
                in_frame = cv2.resize(frame, (w, h))
                in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                in_frame = in_frame.reshape((n, c, h, w))
                feed_dict[input_blob] = in_frame
                exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                inf_end = time.time()
                det_time = inf_end - inf_start

                # Parse detection results of the current request
                res = exec_net.requests[cur_request_id].outputs[out_blob]
                detections = list()
                # print([obj for obj in res[0][0] if obj[2]> self.prob_thresh])
                for obj in res[0][0]:
                    # Draw only objects when probability more than specified threshold
                    if obj[2] > self.prob_thresh:
                        detection_data = dict()
                        xmin = int(obj[3] * initial_w)
                        ymin = int(obj[4] * initial_h)
                        xmax = int(obj[5] * initial_w)
                        ymax = int(obj[6] * initial_h)
                        class_id = int(obj[1])
                        detection_data['class'] = class_id
                        detection_data['bbox'] = [(xmin, ymin), (xmax, ymax)]
                        detections.append(detection_data)
                        # Draw box and label\class_id
                        color = (min(class_id * 12.5, 255),
                                 min(class_id * 7, 255),
                                 min(class_id * 5, 255))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                      color, 2)
                        det_label = labels_map[class_id] if labels_map else str(class_id)
                        cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                yield(detections)
                # Draw performance stats
                inf_time_message = "Inference time: N\A for async mode" if self.async_mode else \
                    "Inference time: {:.3f} ms".format(det_time * 1000)
                render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
                async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if self.async_mode else \
                    "Async mode is off. Processing request {}".format(cur_request_id)

                cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
                cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (10, 10, 200), 1)

            #
            render_start = time.time()
            cv2.imshow("Detection Results", frame) # Comment this line to stop rendering output
            render_end = time.time()
            render_time = render_end - render_start

            if self.async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                frame = next_frame

            key = cv2.waitKey(1)
            if key == 27:
                break
            if (9 == key):
                self.async_mode = not self.async_mode
                log.info("Switched to {} mode".format("async" if self.async_mode else "sync"))

        cv2.destroyAllWindows()

if __name__=='__main__':

    infer = VinoInfer('/home/saahil/Desktop/saahil/junk/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP32/person-detection-retail-0013.bin', '/home/saahil/Desktop/saahil/junk/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP32/person-detection-retail-0013.xml', 'cam')

    [print(detection) for detection in infer.draw_inference()]
