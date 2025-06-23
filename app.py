import streamlit as st
import cv2
import os
import numpy as np
from openpyxl import Workbook, load_workbook
from datetime import datetime

DATASET_DIR = "datasets"
EXCEL_FILE = "user_details.xlsx"

# Ensure dataset directory exists
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def mse(img1, img2):
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

def save_user_to_excel(name, college, img_path):
    if not os.path.exists(EXCEL_FILE):
        wb = Workbook()
        ws = wb.active
        ws.append(["Name", "College", "ImagePath"])
        wb.save(EXCEL_FILE)
    wb = load_workbook(EXCEL_FILE)
    ws = wb.active
    ws.append([name, college, img_path])
    wb.save(EXCEL_FILE)

def load_users_from_excel():
    users = []
    if not os.path.exists(EXCEL_FILE):
        return users
    wb = load_workbook(EXCEL_FILE)
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and len(row) == 3:
            users.append({"name": row[0], "college": row[1], "img_path": row[2]})
    return users

def register_face():
    st.header("Register Face")
    name = st.text_input("Enter your name")
    college = st.text_input("Enter your college/university")
    poses = ["center", "left", "right", "up", "down"]
    if st.button("Start Registration"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            return
        st.info("Registration started. Please follow the instructions.")
        for pose in poses:
            st.write(f"Please turn your face to the **{pose}**. You have 5 seconds...")
            start_time = datetime.now()
            captured = False
            timeout = 5
            while (datetime.now() - start_time).total_seconds() < timeout and not captured:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Pose: {pose}", use_column_width=True)
                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                    face_img = gray[y:y + h, x:x + w]
                    filename = f"{name.replace(' ', '')}{college.replace(' ', '')}{pose}_{int(datetime.now().timestamp())}.jpg"
                    img_path = os.path.join(DATASET_DIR, filename)
                    cv2.imwrite(img_path, face_img)
                    save_user_to_excel(name, college, img_path)
                    st.success(f"Captured {pose} pose.")
                    captured = True
                if st.button("Cancel Registration", key=f"cancel_{pose}"):
                    cap.release()
                    st.warning("Registration cancelled.")
                    return
            if not captured:
                st.warning(f"No face detected for {pose} pose. Skipping.")
        cap.release()
        st.success("Registration complete!")

def recognize_face():
    st.header("Face Recognition")
    users = load_users_from_excel()
    known_faces = []
    known_names = []
    known_colleges = []
    for user in users:
        if os.path.exists(user["img_path"]):
            img = cv2.imread(user["img_path"], cv2.IMREAD_GRAYSCALE)
            known_faces.append(img)
            known_names.append(user["name"])
            known_colleges.append(user["college"])
    if not known_faces:
        st.warning("No registered faces found. Please register first.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return
    prev_label = None
    st.info("Press 'Stop Recognition' to end.")
    stop = st.button("Stop Recognition", key="stop_recognition")
    frame_placeholder = st.empty()
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read from webcam.")
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
            if min_error < 3000 and matched_index != -1:
                label = f"{known_names[matched_index]} ({known_colleges[matched_index]})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        stop = st.button("Stop Recognition", key="stop_recognition_loop")
    cap.release()
    st.success("Recognition stopped.")