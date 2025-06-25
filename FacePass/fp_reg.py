import cv2
import os
from openpyxl import Workbook, load_workbook
import time

def face_registration(name=None, college=None, dataset_dir="datasets", excel_file="user_details.xlsx", 
                     poses=None, timeout=5, camera_index=0):
   
    if poses is None:
        poses = ["center", "left", "right", "up", "down"]
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    try:
        print("Automatic registration started.")
        
        # Get user input if not provided
        if name is None:
            name = input("Enter your name: ")
        if college is None:
            college = input("Enter your College/University: ")
        
        # Setup or load Excel file
        if not _setup_excel_file(excel_file):
            return False
        
        # Capture faces for each pose
        for pose in poses:
            if not _capture_pose(cap, face_cascade, name, college, pose, dataset_dir, timeout):
                print(f"Failed to capture {pose} pose.")
                return False
        
        # Save user details to Excel
        if not _save_user_details(excel_file, name, college):
            print("Warning: Failed to save user details to Excel.")
        
        print(f"All face angles registered for {name}. Registration complete!")
        return True
        
    except KeyboardInterrupt:
        print("\nRegistration interrupted by user.")
        return False
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

def _setup_excel_file(excel_file):
    """Setup Excel file with headers if it doesn't exist."""
    try:
        if not os.path.exists(excel_file):
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["Name", "College/University"])
            workbook.save(excel_file)
        return True
    except Exception as e:
        print(f"Error setting up Excel file: {str(e)}")
        return False

def _capture_pose(cap, face_cascade, name, college, pose, dataset_dir, timeout):
    """Capture a single pose for the user."""
    print(f"Please turn your face to the {pose}. You have {timeout} seconds...")
    start_time = time.time()
    captured = False
    
    while not captured and (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            return False
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles for all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Show pose and countdown
        cv2.putText(frame, f"Pose: {pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Hold for {max(0, int(timeout - (time.time() - start_time)))}s", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Face Detection", frame)
        
        # Capture after timeout if a face is detected
        if time.time() - start_time >= timeout and len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face_image = frame[y:y + h, x:x + w]
            filename = f"{name.replace(' ', '')}{college.replace(' ', '')}{pose}_{int(time.time())}.jpg"
            filepath = os.path.join(dataset_dir, filename)
            cv2.imwrite(filepath, face_image)
            print(f"Face image ({pose}) saved as {filepath}")
            captured = True
        
        # Allow user to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting early.")
            return False
    
    return captured

def _save_user_details(excel_file, name, college):
    """Save user details to Excel file."""
    try:
        workbook = load_workbook(excel_file)
        sheet = workbook.active
        sheet.append([name, college])
        workbook.save(excel_file)
        return True
    except Exception as e:
        print(f"Error saving to Excel: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
   
    face_registration()
    
    