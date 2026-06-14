# AI Hand Gesture Controlled 3D Interaction System

## Project Overview

This project implements a real-time AI hand gesture recognition and 3D object control system using MediaPipe, SVM, UDP communication, and Unity.

The system captures hand landmarks from a webcam, classifies hand gestures using a trained Support Vector Machine (SVM) model, and sends gesture and motion data to Unity for real-time 3D interaction.

## System Architecture

```text
Webcam
   ↓
MediaPipe Hands
   ↓
21 Hand Landmarks (63 Features)
   ↓
SVM Gesture Classifier
   ↓
UDP Communication
   ↓
Unity
   ↓
3D Object Control
```

## Features

### Hand Landmark Detection

* MediaPipe Hands
* 21 hand landmarks
* 63-dimensional feature vector (x, y, z)

### Gesture Recognition

Three predefined gestures:

| Label | Gesture      |
| ----- | ------------ |
| 0     | Open Hand    |
| 1     | Fist         |
| 2     | Index Finger |

### Motion Tracking

* Index finger tracking
* Exponential Moving Average (EMA) smoothing
* Dead-zone filtering
* Yaw and Pitch calculation

### Unity Integration

* UDP real-time communication
* Hand skeleton visualization
* 3D object rotation control
* Real-time gesture feedback

## Dataset Collection

Hand landmark data are collected using MediaPipe and exported to CSV format.

Each sample contains:

```text
x0,y0,z0
x1,y1,z1
...
x20,y20,z20
```

Total Features:

```text
21 landmarks × 3 coordinates = 63 features
```

## Model Training

### Algorithm

Support Vector Machine (SVM)

### Parameters

```python
kernel='rbf'
C=1.0
gamma='scale'
```

### Training Pipeline

```text
CSV Dataset
    ↓
Feature Extraction
    ↓
Train/Test Split
    ↓
SVM Training
    ↓
Model Evaluation
    ↓
svm_hand_model.pkl
```

## Real-Time Recognition

The trained model predicts hand gestures in real time.

Output Data:

```text
[prediction,
 yaw_angle,
 pitch_angle,
 landmark_coordinates]
```

## Unity Communication

UDP Communication:

```text
IP: 127.0.0.1
Port: 5052
```

Data flow:

```text
Python
   ↓ UDP
Unity Receiver
   ↓
Hand Tracking Script
   ↓
3D Object Rotation
```

## Technologies Used

### Python

* OpenCV
* MediaPipe
* NumPy
* Scikit-Learn
* Joblib

### Unity

* C#
* UDP Socket Communication
* Line Renderer
* Quaternion Rotation

## Project Structure

```text
├── cut2.py
│   └── Collect hand landmark data

├── svm.py
│   └── Train SVM model

├── Recognize.py
│   └── Real-time gesture recognition

├── svm_hand_model.pkl
│   └── Trained model

├── UDPReceive.cs
│   └── UDP receiver

├── HandTracking.cs
│   └── Hand visualization and object control

├── LineCode.cs
│   └── Skeleton connection rendering
```

## Future Improvements

* More gesture classes
* Deep learning models (CNN / LSTM)
* Two-hand tracking
* Dynamic gesture recognition
* VR/AR integration
* Robotic arm control
