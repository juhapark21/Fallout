# MODS 

MODS is a self-defensive transport mechanism for Soup, the Fallout mascot.  

## Features 
- Motorized chair (tank drivetrain) 
- Authentication system via face recognition 
- 2-axis turret system 
- Speech command to fire 

## Dependencies 
pip install: 
- pyserial 
- opencv-python 
- numpy 
- vosk 
- sounddevice 
- onnxruntime 

models: 
- [vosk](https://alphacephei.com/vosk/models) speech recognition model (any works; 40mb for portable)
- [YuNet](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) (face detection) 
- [SFace](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) (face recognition) 

## Why 
Soup needs to survive in the upcoming apocalypse and needs a way to move since Soup lacks both arms and legs. 

We originally meant for this device to be a human-ridable vehicle but due to torque constraints on the motor we were only able to seat Soup. 

## Demo video 
[![Demo video](https://img.youtube.com/vi/9z48i6hfK3s/0.jpg)](https://youtu.be/9z48i6hfK3s) 

## Zine 
<img width="1260" height="1785" alt="zine" src="https://github.com/user-attachments/assets/59f77a40-d95d-447e-893e-919e8e2f532d" />


