# Face Pass: An AI-enabled Facial Recognition and Access Control System

*Face Pass* is a facial recognition–based attendance and access control system built using Python and OpenCV. It captures facial images in different poses, recognizes individuals in real-time using webcam input, and logs attendance securely — all without requiring internet or third-party APIs.

---

## Features

- Face registration with 5 head poses: center, left, right, up, down
- Real-time face recognition using Mean Squared Error (MSE) comparison
- Excel-based user registration (user_details.xlsx)
- CSV-based attendance logging with timestamps (attendance.csv)
- Offline, lightweight system — no cloud or API dependencies
- Simple and clean UI using OpenCV windows

---

## Project Structure

facepass-opencv/ ├── datasets/                # Stored face images ├── attendance.csv           # Attendance log file ├── user_details.xlsx        # Registered users' data ├── face_register.py         # Script for user registration ├── face_recognize.py        # Script for real-time recognition └── README.md

---

## Requirements

- Python 3.8 or above
- OpenCV
- NumPy
- openpyxl

### Installation

```bash
pip install opencv-python numpy openpyxl


---

How to Use

1. Register a User

python face_register.py

Enter name and college when prompted.

Follow on-screen pose instructions.

Face images will be saved in the datasets/ folder.


2. Start Recognition and Attendance

python face_recognize.py

Recognizes faces in real-time via webcam.

Displays name, college, and confidence score.

Logs attendance in attendance.csv.


Press q to quit.


---

File Formats

Image Filenames

Captured face images are saved as:

NameCollegePose_Timestamp.jpg
Example: TanishaNITAcenter_1720123456.jpg

Attendance Log (CSV)

Each recognition entry is saved as:

Name, College, Timestamp
