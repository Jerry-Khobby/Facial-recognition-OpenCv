import cv2
import os
import csv
from datetime import datetime
import face_recognition
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_known_encodings(known_faces_folder):
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use face_recognition to compute face encodings
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding)>0:
                face_encoding = face_encoding[0]
                known_encodings.append(face_encoding)
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"No faces found in {filename}")
                

    return known_encodings, known_names




def recognize_faces(video_capture, known_encodings, known_names):
    marked_attendance = set()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Perform face alignment if needed
            
            # Recognize the face
            face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
            
            # Compare the face encoding with known encodings
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            
            if min_distance < 0.6:  # Adjust this threshold as needed
                name = known_names[min_distance_index]
                
                if name not in marked_attendance:
                    mark_attendance(name)
                    marked_attendance.add(name)
                else:
                    print(f"Attendance already marked for today: {name}")
            else:
                print("Unknown person. Please register first.")
                name = "Unknown"
                
            # Draw rectangle around the face and display name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def mark_attendance(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if name != "Unknown":
        # Check if the file exists, if not, create it
        file_exists = os.path.isfile('attendance.csv')
        
        # Check if the name is already in the attendance file
        already_marked = False
        if os.path.isfile('attendance.csv'):
            with open('attendance.csv', mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['Name'] == name:
                        already_marked = True
                        break
        
        if not already_marked:
            with open('attendance.csv', mode='a', newline='') as file:
                fieldnames = ['Name', 'Time']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({'Name': name, 'Time': timestamp})
                
            print(f"Attendance marked for: {name} at {timestamp}")
        else:
            print(f"Attendance already marked for: {name}")
    else:
        print("Attendance not marked for unknown person.")

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)  # Use the camera (0 or 1) or video file path
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load known face encodings and names from a folder
    known_faces_folder = "./facesData"
    known_encodings, known_names = load_known_encodings(known_faces_folder)
    
    # Start face recognition in real-time
    recognize_faces(video_capture, known_encodings, known_names)
