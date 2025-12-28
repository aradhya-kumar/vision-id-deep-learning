import cv2
import os
import numpy as np
from PIL import Image
import pickle  # to save name-ID mappings

# Base directories
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "dataset")
trainer_path = os.path.join(base_dir, "trainer")
os.makedirs(trainer_path, exist_ok=True)

# Haar Cascade for detection
face_cascade = cv2.CascadeClassifier(os.path.join(base_dir, "haarcascade_frontalface_default.xml"))
if face_cascade.empty():
    print("[ERROR] Haar Cascade not loaded! Check haarcascade_frontalface_default.xml path.")
    exit()

# LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Label encoding variables
current_id = 0
label_ids = {}
faces = []
ids = []

print("[INFO] Scanning dataset and encoding faces...")

# Walk through all folders inside dataset/
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).split("_")[0] # folder name = person's name

            # Assign a numeric ID to each unique person
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            # Load image and convert to grayscale
            image = Image.open(path).convert("L")  # grayscale
            image_np = np.array(image, "uint8")

            # Detect faces in the image
            faces_detected = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces_detected:
                roi = image_np[y:y+h, x:x+w]
                faces.append(roi)
                ids.append(id_)

print(f"[INFO] Found {len(label_ids)} unique persons.")
print("[INFO] Training the recognizer...")

# Train the recognizer
recognizer.train(faces, np.array(ids))

# Save trained model
trainer_file = os.path.join(trainer_path, "trainer.yml")
recognizer.save(trainer_file)

# Save name-ID mapping
with open(os.path.join(trainer_path, "labels.pickle"), "wb") as f:
    pickle.dump(label_ids, f)

print("\n[INFO] Training complete!")
print(f"[INFO] Model saved at: {trainer_file}")
print(f"[INFO] Labels saved at: {os.path.join(trainer_path, 'labels.pickle')}")
