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
    marked_attendance = set()  # To store names for which attendance is already marked
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
            
            # Recognize the face
            face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
            # Compare the face encoding with known encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
                
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
                
                
                if name not in marked_attendance:
                    #Mark attendance only if not already marked 
                    mark_attendance(name)
                    marked_attendance.add(name)
                else:
                    print(f"Attendance already marked for today:{name}")
            else:
                print("Unknown person.Please register first")
                
                
                #print a message for unknown faces 
                cv2.putText(frame,"Unknown Person",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)
                
            # Draw rectangle around the face and display name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
            # Mark attendance
            mark_attendance(name)
        
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
    
    # Load known face encodings and names from a folder
    known_faces_folder = "./Jeremiah Anku Coblah"
    known_encodings, known_names = load_known_encodings(known_faces_folder)
    
    # Start face recognition in real-time
    recognize_faces(video_capture, known_encodings, known_names)
