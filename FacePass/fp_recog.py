import cv2
import os
import numpy as np
import re

# Path to your datasets folder
DATASET_DIR = "datasets"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper: Calculate MSE between two images
def mse(img1, img2):
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

# Load all known faces from datasets folder
known_faces = []
known_names = []
known_colleges = []

for filename in os.listdir(DATASET_DIR):
    if filename.endswith(".jpg"):
        # Extract name and college from filename
        match = re.match(r"([A-Za-z]+)([A-Za-z]+)(center|left|right|up|down)_\d+\.jpg", filename)
        if match:
            name = match.group(1)
            college = match.group(2)
            img_path = os.path.join(DATASET_DIR, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                face_img = gray[y:y + h, x:x + w]
                known_faces.append(face_img)
                known_names.append(name)
                known_colleges.append(college)
            else:
                print(f"No face detected in {filename}")

print(f"Loaded {len(known_faces)} known faces.")

if len(known_faces) == 0:
    print("No known faces loaded. Please check your dataset folder and filename format.")
    exit()

# Start webcam recognition
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        min_error = float('inf')
        matched_index = -1

        for idx, known in enumerate(known_faces):
            error = mse(face_img, known)
            if error < min_error:
                min_error = error
                matched_index = idx

        # Adjust threshold as needed
        if min_error < 3000 and matched_index != -1:
            label = f"{known_names[matched_index]} ({known_colleges[matched_index]})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()