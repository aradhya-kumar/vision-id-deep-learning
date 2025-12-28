import cv2
import os
import pickle

# Base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
trainer_dir = os.path.join(base_dir, "trainer")

# Load Haar Cascade for detection
face_cascade = cv2.CascadeClassifier(os.path.join(base_dir, "haarcascade_frontalface_default.xml"))
if face_cascade.empty():
    print("[ERROR] Haar Cascade not loaded! Check haarcascade_frontalface_default.xml path.")
    exit()

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = os.path.join(trainer_dir, "trainer.yml")

if not os.path.exists(model_path):
    print("[ERROR] Trained model not found! Run encode_faces.py first.")
    exit()
    

recognizer.read(model_path)

# Load label (ID → name) mapping
labels_path = os.path.join(trainer_dir, "labels.pickle")
if not os.path.exists(labels_path):
    print("[ERROR] labels.pickle not found! Run encode_faces.py first.")
    exit()

with open(labels_path, "rb") as f:
    label_ids = pickle.load(f)

# Invert mapping: {name: id} → {id: name}
labels = {v: k for k, v in label_ids.items()}

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Cannot access the webcam.")
    exit()

print("[INFO] Starting real-time face recognition...")
print("[INFO] Press 'ESC' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # LBPH gives lower confidence = better match
        if confidence < 70:  # adjust threshold if needed
            name = labels.get(id_, "Unknown")
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        # Draw rectangle & info
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}", (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"{confidence_text}", (x+5, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
print("[INFO] Face recognition stopped.")
