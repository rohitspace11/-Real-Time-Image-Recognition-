Real-Time-Image-Recognition-

Description

This project implements a real-time image recognition system using the pre-trained MobileNetV2 model. The system uses a webcam to capture live video frames, processes each frame, and classifies the objects in real time. The predicted label and its confidence score are displayed on the video feed.

MobileNetV2 is a lightweight, efficient convolutional neural network architecture pre-trained on the ImageNet dataset. It can classify images into 1,000 different categories, including everyday objects, animals, tools, and more.

Features

Real-Time Processing: Captures video feed from a webcam and performs object recognition frame by frame.

Pre-Trained Model: Uses MobileNetV2 trained on ImageNet for robust and accurate predictions.

Confidence Score: Displays the model's confidence percentage alongside the predicted label.

Ease of Use: Simple implementation with clear output.

Prerequisites

Ensure you have the following installed on your system:

Python 3.7+

TensorFlow

OpenCV

NumPy

To install the required libraries, run:

pip install tensorflow opencv-python numpy

How to Run

Clone the repository or copy the code.

Save the script as real_time_image_recognition.py.

Run the script using Python:

python real_time_image_recognition.py

Allow camera access when prompted.

Press q to exit the program.

How It Works

Video Capture: The program uses OpenCV to access the webcam and capture video frames.

Frame Preprocessing:

Frames are resized to the input size required by MobileNetV2 (224x224 pixels).

Pixel values are normalized using preprocess_input from TensorFlow.

Prediction:

The processed frame is passed through MobileNetV2.

The model outputs probabilities for 1,000 categories.

The top prediction is decoded and displayed.

Display:

The predicted label and confidence score are overlaid on the video feed.

The processed video feed is displayed in a window.

Example Output

While running, the program will display the webcam feed with labels and confidence scores for the recognized objects. For example:

Label: "Laptop"

Confidence: "98.45%"

Applications

Object recognition in real time.

Educational purposes to understand deep learning in computer vision.

Basis for custom real-time classification projects (e.g., face mask detection).

Future Enhancements

Custom Classes: Fine-tune MobileNetV2 on a custom dataset for specific use cases like face mask detection or industrial object recognition.

Performance Optimization: Use TensorFlow Lite for faster inference on edge devices.

Multi-Object Detection: Extend the project to detect and localize multiple objects in a single frame using YOLO or SSD.


