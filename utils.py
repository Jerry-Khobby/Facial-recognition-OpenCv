import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Function to load the dataset
def load_dataset(dataset_path):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for filename in image_files:
        # Extract label (student's name or ID) from the filename
        label = os.path.splitext(filename)[0]
        
        if label not in label_dict:
            label_dict[label] = current_label
            current_label += 1
        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply image preprocessing techniques (e.g., histogram equalization)
        gray = cv2.equalizeHist(gray)
        
        # Resize images to a standard size
        gray = cv2.resize(gray, (100, 100))
        
        faces.append(gray)
        labels.append(label_dict[label])

    # Save the label dictionary to a file
    with open("label_dict.txt", "w") as file:
        for key, value in label_dict.items():
            file.write(f"{key},{value}\n")

    return faces, np.array(labels)

# Function to train the recognizer
def train_recognizer(dataset_path):
    faces, labels = load_dataset(dataset_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

    # Choose the recognizer algorithm (e.g., LBPH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer on the training set
    recognizer.train(X_train, y_train)
    
    # Evaluate the recognizer on the testing set
    total = 0
    correct = 0
    for i in range(len(X_test)):
        label_predicted, _ = recognizer.predict(X_test[i])
        if label_predicted == y_test[i]:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Save the trained model
    recognizer.save("trained_model.yml")

    print("Training complete.")

if __name__ == "__main__":
    dataset_path = "./Humans"  # Update the path to your folder
    train_recognizer(dataset_path)
