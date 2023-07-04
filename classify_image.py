import numpy as np
from tensorflow import keras
from PIL import Image

# Define constants
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

# Load and preprocess a single image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = image.convert("L")  # Convert to grayscale
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 255.0
    image_array = image_array.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
    return image_array

# Define the path to the image you want to classify
image_path = "/Users/jaggehns/Desktop/image-classification-monolith/images.jpeg"

# Load the pre-trained model
model = keras.models.load_model("/Users/jaggehns/Desktop/image-classification-monolith/src/fashion_mnist_model.h5")

# Load and preprocess the image
image_array = load_image(image_path)

# Perform prediction on the image
prediction = model.predict(image_array)

# Get the predicted label
predicted_label = np.argmax(prediction)

# Get the predicted clothing type
predicted_clothing_type = clothing_types[predicted_label]

# Print the predicted label and clothing type
print(f"Predicted Label: {predicted_label}")
print(f"Predicted Clothing Type: {predicted_clothing_type}")
