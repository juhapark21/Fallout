#!/usr/bin/env bash
# Downloads the pretrained YuNet (detection) and SFace (recognition) ONNX
# models from OpenCV's official model zoo into ./models/
set -e

mkdir -p models

echo "Downloading YuNet face detector..."
curl -fsSL -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

echo "Downloading SFace face recognizer..."
curl -fsSL -o models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

echo "Done. Models saved to ./models/"
