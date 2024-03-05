import cv2
import matplotlib.pyplot as plt

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

def detect_faces_webcam(scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
    video_capture = cv2.VideoCapture(0)  # Use webcam (change the index if needed)

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

def main(input_path, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        detect_faces_image(input_path, scaleFactor, minNeighbors, minSize)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        detect_faces_video(input_path, scaleFactor, minNeighbors, minSize)
    elif input_path.lower() == 'webcam':
        detect_faces_webcam(scaleFactor, minNeighbors, minSize)
    else:
        print("Unsupported input")

if __name__ == "__main__":
    input_path = "webcam"  # Change this to the path of your image, video file, or "webcam"
    main(input_path, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
