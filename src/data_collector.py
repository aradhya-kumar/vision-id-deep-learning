import cv2
import os

# Base directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')

# Ensure dataset folder exists
os.makedirs(dataset_dir, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Check if the Haar Cascade loaded correctly
if face_cascade.empty():
    print("[ERROR] Haar cascade file not loaded! Check haarcascade_frontalface_default.xml path.")
    exit()

# Ask for user name and numeric ID
user_name = input("Enter User Name: ").strip()
user_id = input("Enter Numeric User ID (e.g., 1, 2, 3): ").strip()

# Create a folder for each user
user_folder = os.path.join(dataset_dir, f"{user_name}_{user_id}")
os.makedirs(user_folder, exist_ok=True)

# Initialize camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Cannot access the webcam.")
    exit()

print(f"[INFO] Collecting face data for {user_name} (ID: {user_id})...")
count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]

        # Save face image with consistent naming
        filename = os.path.join(user_folder, f"User.{user_id}.{count}.jpg")
        cv2.imwrite(filename, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Image {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Collecting Faces', frame)

    # Press ESC or collect 100 images
    if cv2.waitKey(1) == 27 or count >= 100:
        break

cam.release()
cv2.destroyAllWindows()

print(f"\n[INFO] {count} images of {user_name} collected successfully.")
print(f"[INFO] Saved at: {user_folder}")
