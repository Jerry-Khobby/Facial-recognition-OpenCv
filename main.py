import cv2
import matplotlib.pyplot as plt
import psycopg2
import numpy as np
import face_recognition



#first things, connect with the database 
def connect_db():
    conn=psycopg2.connect(
        dbname='STUDENTS',
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )
    return conn


# Function to compare detected face with faces in the database
def compare_faces(detected_face_encoding):
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Retrieve student names and face encodings from the database
        cur.execute("SELECT name, passport_image, face_encoding FROM student_passports")
        rows = cur.fetchall()

        for name, passport_image, face_encoding_bytes in rows:
            db_face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float64)
            db_face_encoding = db_face_encoding.reshape((128,))

            # Compare the detected face with each face in the database
            results = face_recognition.compare_faces([db_face_encoding], detected_face_encoding)
            if results[0]:
                return name  # Return the name of the matching student

        return "Unknown"  # Return "Unknown" if no match found
    except psycopg2.Error as e:
        print("Error comparing faces:", e)
        return "Unknown"
    finally:
        cur.close()
        conn.close()

# Function to detect faces using webcam

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces_image(image_path, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def detect_faces_video(video_path, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
    video_capture = cv2.VideoCapture(video_path)

    while True:
        result, video_frame = video_capture.read()
        if not result:
            break

        gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

        for (x, y, w, h) in faces:
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.imshow("My Face Detection Project", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to detect faces using webcam
def detect_faces_webcam(scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
    video_capture = cv2.VideoCapture(0)  # Use webcam (change the index if needed)

    while True:
        result, video_frame = video_capture.read()
        if not result:
            break

        gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

        for (x, y, w, h) in faces:
            detected_face = video_frame[y:y+h, x:x+w]
            
            # Convert BGR to RGB
            detected_face_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
            
            detected_face_encoding = face_recognition.face_encodings(detected_face_rgb)
            
            if len(detected_face_encoding) > 0:
                name = compare_faces(detected_face_encoding[0])
                cv2.putText(video_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            else:
                cv2.putText(video_frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        cv2.imshow("My Face Detection Project", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()



def main(input_path, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        detect_faces_image(input_path, scaleFactor, minNeighbors, minSize)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        detect_faces_video(input_path, scaleFactor, minNeighbors, minSize)
    elif input_path.lower() == 'webcam':
        detect_faces_webcam(scaleFactor, minNeighbors, minSize)
            # Assuming detected_face_encoding is the detected face encoding
        detected_face_encoding = np.random.rand(128)  # Example detected face encoding
        name = compare_faces(detected_face_encoding)
        print("Matched Student Name:", name)
    else:
        print("Unsupported input")

if __name__ == "__main__":
    input_path = "webcam"  # Change this to the path of your image, video file, or "webcam"
    main(input_path, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
