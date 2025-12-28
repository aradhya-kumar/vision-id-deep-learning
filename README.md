# 👁️ Vision ID — Face Recognition System

Vision ID is a computer vision–based face recognition system built using
Python and OpenCV. The system detects human faces, encodes facial features,
and recognizes known individuals in real time using a webcam.

---

## 🔍 Overview

The project follows a complete face recognition pipeline:
- Face data collection
- Face encoding
- Real-time face recognition

It uses classical computer vision techniques combined with facial feature
encoding to identify individuals accurately.

---

## 🧠 How It Works

1. **Data Collection**
   - Captures face images using a webcam
   - Detects faces using Haar Cascade

2. **Face Encoding**
   - Extracts facial embeddings
   - Stores encodings for recognition

3. **Face Recognition**
   - Matches live webcam input against stored encodings
   - Displays recognized names in real time

---

## 🛠 Technologies Used

- Python
- OpenCV
- face_recognition
- NumPy
- Haar Cascade Classifier

---

## 📁 Project Structure

```text
src/
├── data_collector.py                    # Collection and Storing Data
├── encodingfaces.py                     # Encoding the collected Data and Trains Model
└── recognize_faces.py                   # Live camera to detect faces
modules/
└── haarcascade_frontalface_default.xml  # Used for detecting faces
```
---

## ⚙️ How It Works

1. The system captures images or video frames using a webcam
2. Faces are detected in each frame using a Haar Cascade classifier
3. Facial features are extracted and converted into numerical encodings
4. Known face encodings are loaded from the stored dataset
5. The detected face encodings are compared with known encodings
6. If a match is found, the person is identified; otherwise, the face is marked as unknown

---

## ▶️ How to Run (Face Recognition System)

1. Clone the repository
    git clone https://github.com/aradhya-kumar/vision-id-face-recognition.git
2. Install required dependencies
    pip install opencv-python face-recognition numpy
3. Ensure the following files are present:
    haarcascade_frontalface_default.xml
    encodings.pkl
4. Run the face recognition script
    python recognize_faces.py
