# Drowsiness-Detection-system
# Creating the README.md file for the Drowsiness Detection System

readme_content = """
# Drowsiness Detection System

## Overview
This project is a real-time drowsiness detection system that uses computer vision and machine learning techniques to monitor eye movements and detect signs of drowsiness. It leverages OpenCV for video capture and image processing, Dlib for facial landmark detection, and SciPy for calculating the Eye Aspect Ratio (EAR). If the system detects prolonged eye closure, an alert sound is triggered to prevent accidents due to driver fatigue. This project is ideal for enhancing safety in automotive applications and other scenarios requiring real-time drowsiness monitoring.

---

## Features
- Real-time video capture and face detection.
- Accurate eye landmark detection using Dlib's pre-trained model.
- Calculation of Eye Aspect Ratio (EAR) to monitor eye closure.
- Audio alert when drowsiness is detected.
- Lightweight implementation suitable for laptops and embedded devices.

---

## Technologies Used
- **Python**
- **OpenCV**: For video capture and image processing.
- **Dlib**: For face and facial landmark detection.
- **SciPy**: For calculating the Euclidean distance.
- **Imutils**: For resizing and manipulating images.
- **Playsound**: For triggering alert sounds.
- **Threading**: For non-blocking alert sound execution.

---

## System Architecture and Flowchart
The flow of the system is as follows:
1. **Video Capture**: Using OpenCV to get real-time video frames.
2. **Face Detection**: Using Dlib's frontal face detector.
3. **Eye Landmark Detection**: Extracts coordinates for left and right eyes.
4. **EAR Calculation**: Calculates the Eye Aspect Ratio for both eyes.
5. **Drowsiness Detection Logic**:
   - If EAR is below a certain threshold for a specified duration, the system detects drowsiness.
   - An alert sound is triggered to wake up the user.
6. **Reset and Repeat**: The timer resets if the eyes are open, and the process repeats for every frame.

You can download the flowchart from [here](sandbox:/mnt/data/drowsiness_detection_flowchart.png).

---

## Industry and Community Applications
1. **Automotive Safety**: Prevents accidents by monitoring driver fatigue.
2. **Workplace Safety**: Monitors heavy machinery operators for drowsiness.
3. **Healthcare**: Can be adapted for sleep disorder detection.

---

## Installation and Setup
1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/drowsiness-detection.git
    cd drowsiness-detection
    ```

2. **Install Dependencies**
    Make sure you have Python and pip installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Dlib Model**
    - Download the `shape_predictor_68_face_landmarks.dat` file from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
    - Extract the `.bz2` file and place it in the `models/` folder.

---

## How to Run
```bash
python drowsiness_detection.py
