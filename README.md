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
├── data_collector.py            # Application entry point
├── encodingfaces.py             # Core chatbot logic
└──recognize_faces.py           # Stores questions & answers 
```
---
