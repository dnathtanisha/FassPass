import cv2
import os
import numpy as np
import re

# Settings
DATASET_DIR = "datasets"
MSE_THRESHOLD = 3000  # Error threshold for matching
CONFIDENCE_THRESHOLD = 60  # Minimum match confidence in %

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Compute Mean Squared Error
def mse(img1, img2):
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

# Load known faces from dataset folder
def load_known_faces():
    known_faces = []
    known_names = []
    known_colleges = []

    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg"):
            match = re.match(r"([A-Za-z]+)([A-Za-z]+)(center|left|right|up|down)_\d+\.jpg", filename)
            if match:
                name = match.group(1)
                college = match.group(2)
                path = os.path.join(DATASET_DIR, filename)

                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                    face_img = gray[y:y + h, x:x + w]
                    known_faces.append(face_img)
                    known_names.append(name)
                    known_colleges.append(college)

    return known_faces, known_names, known_colleges

# Calculate confidence from MSE
def mse_to_confidence(error):
    return max(0, 100 - (error / MSE_THRESHOLD) * 100)

def recognize_faces():
    known_faces, known_names, known_colleges = load_known_faces()

    if not known_faces:
        print("❌ No known faces found. Run registration first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("📹 Starting recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))

            min_error = float("inf")
            best_index = -1

            for i, known_face in enumerate(known_faces):
                known_resized = cv2.resize(known_face, (100, 100))
                error = mse(face_img, known_resized)

                if error < min_error:
                    min_error = error
                    best_index = i

            confidence = mse_to_confidence(min_error)

            if confidence >= CONFIDENCE_THRESHOLD:
                name = known_names[best_index]
                college = known_colleges[best_index]
                label = f"{name} ({int(confidence)}%)"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                college = ""
                color = (0, 0, 255)

            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 50), (x + w, y), color, -1)

            # Text: Name and college
            cv2.putText(frame, label, (x + 5, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if college:
                cv2.putText(frame, college, (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition (OpenCV)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Recognition ended.")

if __name__ == "__main__":
    recognize_faces()
