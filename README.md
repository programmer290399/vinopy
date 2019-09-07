# vinopy
A python API for using OpenVINO 

## How to setup :
1. You must have OpenVINO installed on your machine , you can find the installation instructions [here](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started)
2. Clone the repo and install dependencies (make sure you're using python 3.5+):
    ```
    git clone https://github.com/programmer290399/vinopy.git
    cd vinopy
    pip install -r requirements.txt
    ```

## How to use :

```py
    from vinopy import VinoInfer

    infer_handle = VinoInfer('Input_stream (can be rtsp/cam/path to video_file/image)')
    # for video stream 
    infer_handle.draw_inference_from_video()
```