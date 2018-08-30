# InVideo Object-Detector
Specify model and video file to run object detection. Useful for testing object detection models.

## Prerequisites

This small python3 program makes heavy use of **OpenCv** library,  **PyQt5** and **TensorFlow** frameworks. 
You can install PyQt5 and TensorFlow by running.
 ```bash
 pip install pyqt5
 pip install tensorflow (or tensorflow-gpu, if you have an compatible Nvidia GPU)
 ```
The opencv-python package offered through the pip package manager doesn`t include the necessary video processing libraries, you will need to build OpenCV for yourself.

The steps on building OpenCV are described well [here](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)

