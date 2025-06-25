import cv2
import os
import numpy as np
import re
from collections import defaultdict

class FaceRecognizer:
    """An improved face recognition system with multiple matching algorithms."""
    
    def __init__(self, dataset_dir="datasets", recognition_threshold=0.6, min_votes=2):
        """
        Initialize the Face Recognizer with improved algorithms.
        
        Args:
            dataset_dir (str): Directory containing face images
            recognition_threshold (float): Threshold for face recognition (0.0-1.0)
            min_votes (int): Minimum votes needed for recognition
        """
        self.dataset_dir = dataset_dir
        self.recognition_threshold = recognition_threshold
        self.min_votes = min_votes
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Multiple face images per person for better accuracy
        self.known_faces = defaultdict(list)  # {person_id: [face_images]} 
        self.person_info = {}  # {person_id: (name, college)}
        
        # Initialize face recognizers
        self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.eigen_recognizer = cv2.face.EigenFaceRecognizer_create()
        self.fisher_recognizer = cv2.face.FisherFaceRecognizer_create()
        
        self.recognizers_trained = False
        
    def preprocess_face(self, face_img, target_size=(100, 100)):
        """
        Preprocess face image for better recognition.
        
        Args:
            face_img (numpy.ndarray): Input face image
            target_size (tuple): Target size for resizing
            
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        # Resize to standard size
        face_img = cv2.resize(face_img, target_size)
        
        # Histogram equalization for better contrast
        face_img = cv2.equalizeHist(face_img)
        
        # Gaussian blur to reduce noise
        face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
        
        # Normalize pixel values
        face_img = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)
        
        return face_img
    
    def calculate_similarity_metrics(self, face1, face2):
        """
        Calculate multiple similarity metrics between two faces.
        
        Args:
            face1, face2 (numpy.ndarray): Face images to compare
            
        Returns:
            dict: Dictionary of similarity scores
        """
        face1 = self.preprocess_face(face1)
        face2 = self.preprocess_face(face2)
        
        # Template matching
        result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
        template_score = np.max(result)
        
        # Histogram comparison
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Structural similarity (simplified)
        mse = np.mean((face1.astype(float) - face2.astype(float)) ** 2)
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse)) if mse > 0 else 100
        psnr_score = min(psnr / 100, 1.0)  # Normalize to 0-1
        
        return {
            'template': template_score,
            'histogram': hist_score,
            'psnr': psnr_score
        }
    
    def load_known_faces(self):
        """
        Load all known faces from the dataset directory with improved organization.
        
        Returns:
            bool: True if faces loaded successfully, False otherwise
        """
        if not os.path.exists(self.dataset_dir):
            print(f"Dataset directory '{self.dataset_dir}' does not exist.")
            return False
        
        self.known_faces.clear()
        self.person_info.clear()
        
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".jpg"):
                # Extract name and college from filename
                match = re.match(r"([A-Za-z]+)([A-Za-z]+)(center|left|right|up|down)_\d+\.jpg", filename)
                if match:
                    name = match.group(1)
                    college = match.group(2)
                    person_id = f"{name}_{college}"
                    
                    img_path = os.path.join(self.dataset_dir, filename)
                    
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Could not load image: {filename}")
                            continue
                            
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        if len(faces) > 0:
                            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                            face_img = gray[y:y + h, x:x + w]
                            face_img = self.preprocess_face(face_img)
                            
                            self.known_faces[person_id].append(face_img)
                            self.person_info[person_id] = (name, college)
                        else:
                            print(f"No face detected in {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
        
        total_faces = sum(len(faces) for faces in self.known_faces.values())
        print(f"Loaded {total_faces} face images for {len(self.known_faces)} people.")
        
        # Train recognizers if we have enough data
        if len(self.known_faces) >= 2:
            self.train_recognizers()
        
        return len(self.known_faces) > 0
    
    def train_recognizers(self):
        """Train the face recognizers with loaded data."""
        try:
            faces = []
            labels = []
            label_to_person = {}
            
            for label, (person_id, face_list) in enumerate(self.known_faces.items()):
                label_to_person[label] = person_id
                for face in face_list:
                    faces.append(face)
                    labels.append(label)
            
            if len(faces) < 2:
                return
            
            faces = np.array(faces)
            labels = np.array(labels)
            
            # Train LBPH recognizer (works best with few samples)
            self.lbph_recognizer.train(faces, labels)
            
            # Train other recognizers if we have enough samples
            if len(faces) >= 10:
                # Flatten faces for Eigen and Fisher recognizers
                faces_flat = faces.reshape(len(faces), -1)
                
                try:
                    self.eigen_recognizer.train(faces, labels)
                    self.fisher_recognizer.train(faces, labels)
                except Exception as e:
                    print(f"Warning: Could not train Eigen/Fisher recognizers: {e}")
            
            self.label_to_person = label_to_person
            self.recognizers_trained = True
            print("Face recognizers trained successfully.")
            
        except Exception as e:
            print(f"Error training recognizers: {e}")
    
    def recognize_face(self, face_img):
        """
        Recognize a face using multiple algorithms and voting.
        
        Args:
            face_img (numpy.ndarray): Face image to recognize
            
        Returns:
            tuple: (name, college, confidence) or (None, None, None) if not recognized
        """
        if len(self.known_faces) == 0:
            return None, None, None
        
        face_img = self.preprocess_face(face_img)
        votes = defaultdict(float)
        
        # Method 1: Template matching with multiple similarity metrics
        for person_id, face_list in self.known_faces.items():
            person_scores = []
            
            for known_face in face_list:
                metrics = self.calculate_similarity_metrics(face_img, known_face)
                # Weighted combination of metrics
                combined_score = (
                    metrics['template'] * 0.4 +
                    metrics['histogram'] * 0.3 +
                    metrics['psnr'] * 0.3
                )
                person_scores.append(combined_score)
            
            # Use the best score for this person
            best_score = max(person_scores)
            if best_score > self.recognition_threshold:
                votes[person_id] += best_score
        
        # Method 2: LBPH recognizer (if trained)
        if self.recognizers_trained:
            try:
                label, confidence = self.lbph_recognizer.predict(face_img)
                # LBPH gives lower confidence for better matches
                if confidence < 80:  # Threshold for LBPH
                    person_id = self.label_to_person.get(label)
                    if person_id:
                        # Convert confidence to similarity score
                        similarity = max(0, (100 - confidence) / 100)
                        votes[person_id] += similarity * 0.8
            except Exception as e:
                pass
        
        # Find the person with highest votes
        if votes:
            best_person = max(votes, key=votes.get)
            best_score = votes[best_person]
            
            # Require minimum votes and confidence
            if best_score >= self.min_votes * self.recognition_threshold:
                name, college = self.person_info[best_person]
                return name, college, best_score
        
        return None, None, 0.0

def face_recognition(dataset_dir="datasets", recognition_threshold=0.6, camera_index=0, 
                    show_confidence=False, window_name="Face Recognition", min_votes=1):
    """
    Start real-time face recognition using webcam with improved accuracy.
    
    Args:
        dataset_dir (str): Directory containing face images
        recognition_threshold (float): Similarity threshold for recognition (0.0-1.0)
        camera_index (int): Camera index for cv2.VideoCapture
        show_confidence (bool): Whether to show confidence scores
        window_name (str): Name of the display window
        min_votes (int): Minimum votes needed for recognition
        
    Returns:
        bool: True if ran successfully, False if error occurred
    """
    # Initialize face recognizer
    recognizer = FaceRecognizer(dataset_dir, recognition_threshold, min_votes)
    
    # Load known faces
    if not recognizer.load_known_faces():
        print("No known faces loaded. Please check your dataset folder and filename format.")
        return False
    
    # Start webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Improved face recognition started. Press 'q' to quit.")
    print(f"Recognition threshold: {recognition_threshold}")
    print(f"Minimum votes required: {min_votes}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam.")
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with better parameters
            faces = recognizer.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                # Add padding around face
                padding = 10
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
                h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
                
                face_img = gray[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
                name, college, confidence = recognizer.recognize_face(face_img)
                
                if name and college:
                    label = f"{name} ({college})"
                    if show_confidence:
                        label += f" [{confidence:.2f}]"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = "Unknown"
                    if show_confidence and confidence > 0:
                        label += f" [{confidence:.2f}]"
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nFace recognition interrupted by user.")
    except Exception as e:
        print(f"Error during face recognition: {str(e)}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True

def recognize_single_image(image_path, dataset_dir="datasets", recognition_threshold=0.6):
    
    recognizer = FaceRecognizer(dataset_dir, recognition_threshold)
    
    if not recognizer.load_known_faces():
        print("No known faces loaded.")
        return []
    
    # Load and process the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = recognizer.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    
    results = []
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        name, college, confidence = recognizer.recognize_face(face_img)
        results.append((name, college, confidence, (x, y, w, h)))
    
    return results

# Example usage
if __name__ == "__main__":
   
    face_recognition()
    
    
    