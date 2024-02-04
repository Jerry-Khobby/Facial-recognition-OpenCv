import cv2
import matplotlib.pyplot as plt


imagePath="myimage.jpg"
img=cv2.imread(imagePath)
print(img.shape)

""" 
The arrays values represent the pictures height, width, and channels respectively. Since this is a color image, there are three channels used to depict it - blue, green, and red (BGR).  """


#convert the image to grayscale,before performing the facial detections on it 
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

#the next step is to load the classifiers 
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#perfom the face detections 
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

print("Detected faces:", face)

#drawing a bounding box 
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

#display the image 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#I have to use the matplotlib library to display the image 
plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
