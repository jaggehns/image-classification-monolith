import csv
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define constants
TEST_DATA_PATH = "/Users/jaggehns/Desktop/image-classification-monolith/src/fashion-mnist_test.csv"
NUM_CLASSES = 10  # Number of clothing classes
IMAGE_SIZE = 28  # Size of the images

clothing_types = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Load test data from CSV file
def load_test_data():
    x_test = []
    y_test = []

    with open(TEST_DATA_PATH, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            label = int(row[0])
            pixels = np.array(row[1:], dtype=np.float32) / 255.0
            image = pixels.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
            x_test.append(image)
            y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_test, y_test

# Load the test data
x_test, y_test = load_test_data()

# Load the pre-trained model
model = keras.models.load_model("/Users/jaggehns/Desktop/image-classification-monolith/src/fashion_mnist_model.h5")

# Perform prediction on test data
predictions = model.predict(x_test)

# Get the predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Compute accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy}")

# Compute confusion matrix
confusion_mat = confusion_matrix(y_test, predicted_labels)
print("Confusion Matrix:")
print(confusion_mat)

# Generate classification report
classification_rep = classification_report(y_test, predicted_labels, target_names=clothing_types)
print("Classification Report:")
print(classification_rep)

# Print the predicted labels
for i in range(len(x_test)):
    predicted_label = predicted_labels[i]
    actual_label = y_test[i]
    predicted_clothing_type = clothing_types[predicted_label]
    actual_clothing_type = clothing_types[actual_label]
    # print(f"Image {i+1}: Predicted: {predicted_clothing_type}, Actual: {actual_clothing_type}")