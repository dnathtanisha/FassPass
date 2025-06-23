import cv2
import os
from openpyxl import Workbook, load_workbook
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if not os.path.exists("datasets"):
    os.makedirs("datasets")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Automatic registration started.")

Name = input("Enter your name: ")
College = input("Enter your College/University: ")
poses = ["center", "left", "right", "up", "down"]

excel_file = "user_details.xlsx"
if not os.path.exists(excel_file):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Name", "College/University"])  # Remove "Image Path"
    workbook.save(excel_file)
else:
    workbook = load_workbook(excel_file)
    sheet = workbook.active

for pose in poses:
    print(f"Please turn your face to the {pose}. You have 5 seconds...")
    start_time = time.time()
    captured = False
    timeout = 5  # seconds
    while not captured and (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Draw rectangles for all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Show pose and countdown
        cv2.putText(frame, f"Pose: {pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Hold for {max(0, int(5 - (time.time() - start_time)))}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        # Always show the camera window
        cv2.imshow("Face Detection", frame)
        # Capture after 5 seconds if a face is detected
        if time.time() - start_time >= 5 and len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face_image = frame[y:y + h, x:x + w]
            filename = f"{Name.replace(' ', '')}{College.replace(' ', '')}{pose}_{int(time.time())}.jpg"
            filepath = os.path.join("datasets", filename)
            cv2.imwrite(filepath, face_image)
            print(f"Face image ({pose}) saved as {filepath}")
            captured = True
        # Allow user to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting early.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

sheet.append([Name, College])
workbook.save(excel_file)

print(f"All face angles registered for {Name}. Exiting...")
cap.release()
cv2.destroyAllWindows()