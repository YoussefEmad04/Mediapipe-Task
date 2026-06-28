# MediaPipe Vision Studio

A simple Computer Vision web app built with **MediaPipe**, **OpenCV**, and **Streamlit**.

The project combines multiple easy MediaPipe vision tasks in one Streamlit dashboard:

* Face Detection
* Hand Tracking
* Simple Gesture Recognition
* Pose Detection
* Background Removal / Segmentation

The goal of this project is educational: to show how MediaPipe can be used to build practical computer vision applications without training a model from scratch.

---

## Project Demo Idea

The user can upload an image or take a camera snapshot, then choose one of the available vision tasks from the sidebar.
The app processes the image and displays the result.

---

## Project Structure

```text
mediapipe_vision_studio/
│
├── app.py
├── vision_tasks.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tasks Included

### 1. Face Detection

Detects human faces in an image and draws bounding boxes around them.

### 2. Hand Tracking

Detects hands and draws 21 hand landmarks using MediaPipe Hands.

### 3. Gesture Recognition

Uses hand landmarks to recognize simple gestures using rule-based logic.

Supported simple gestures:

* Fist
* Open Palm
* Victory / Peace
* Thumbs Up / Like
* Pointing
* Number of opened fingers

### 4. Pose Detection

Detects human body pose landmarks such as shoulders, elbows, knees, and ankles.

### 5. Background Removal

Uses selfie segmentation to separate the person from the background and blur the background.

---

## Technologies Used

* Python
* MediaPipe
* OpenCV
* NumPy
* Pillow
* Streamlit

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YoussefEmad04/mediapipe_vision_studio.git
cd mediapipe_vision_studio
```

---

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### 3. Install requirements

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```txt
streamlit
mediapipe==0.10.21
opencv-python
numpy<2
Pillow
```

> Note: This project uses the legacy `mp.solutions` API from MediaPipe, so the MediaPipe version is pinned to avoid compatibility issues.

---

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## How It Works

The project is divided into two main files:

### `app.py`

Responsible for the Streamlit user interface.

It allows the user to:

* Upload an image
* Take a camera snapshot
* Choose a vision task
* View the original and processed images

### `vision_tasks.py`

Responsible for the computer vision processing.

Each function takes an RGB image, processes it using MediaPipe, and returns:

1. The processed output image
2. A message describing the result

Example:

```python
output, message = detect_face(image_rgb)
```

---

## Main Functions

| Function                           | Description                                     |
| ---------------------------------- | ----------------------------------------------- |
| `detect_face()`                    | Detects faces and draws bounding boxes          |
| `detect_hands()`                   | Detects hands and draws hand landmarks          |
| `get_finger_states()`              | Checks which fingers are open or closed         |
| `recognize_gesture_from_fingers()` | Converts finger states into a simple gesture    |
| `recognize_gesture()`              | Detects hand landmarks and recognizes a gesture |
| `detect_pose()`                    | Detects body pose landmarks                     |
| `remove_background()`              | Separates the person from the background        |

---

## Educational Notes

This project demonstrates three important Computer Vision concepts:

### Detection

Finding the location of an object.

Example: drawing a bounding box around a face.

### Landmarks

Finding detailed key points on an object.

Example: detecting 21 points on a hand.

### Segmentation

Separating image regions.

Example: separating a person from the background.

---

## Important Note About Gesture Recognition

The gesture recognition part is rule-based.

It does not use a trained gesture classification model.
Instead, it uses hand landmarks and compares finger tip positions with finger joint positions to estimate whether each finger is open or closed.

This is useful for learning and simple demos, but for production-level gesture recognition, a stronger model or more advanced rules may be needed.

---

## Future Improvements

Possible improvements for this project:

* Add real-time webcam support using `streamlit-webrtc`
* Add more hand gestures
* Improve thumb detection for left and right hands
* Add exercise counters using pose landmarks
* Add image download option
* Add custom background replacement instead of blur
* Deploy the app online

---

## Author

Created by **Youssef Emad** as a simple educational MediaPipe and Streamlit computer vision project.
