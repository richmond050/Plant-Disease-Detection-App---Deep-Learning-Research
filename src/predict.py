# src/predict.py

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image

# Constants
MODEL_PATH = Path("models/plant_disease_detector_model.h5")
DATA_DIR = Path("data/raw/PlantVillage")
IMG_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Get and clean class names
def clean_class_name(raw_name: str) -> str:
    """
    Cleans raw folder/class names by removing underscores and duplicates.
    Example: 'Tomato___Tomato_mosaic_virus' ➝ 'Tomato Mosaic Virus'
    """
    name = raw_name.replace("___", " ").replace("_", " ")
    words = name.split()
    cleaned = []
    for word in words:
        if not cleaned or word.lower() != cleaned[-1].lower():
            cleaned.append(word)
    return " ".join(word.capitalize() for word in cleaned)

class_names = sorted([
    folder.name for folder in DATA_DIR.iterdir() if folder.is_dir()
])
class_names = [clean_class_name(name) for name in class_names]

def predict_image(img_path: str) -> str:
    """
    Loads an image, preprocesses it, and makes a prediction using the model.
    Returns a cleaned, human-readable class name.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    print(f"Prediction: {predicted_class}")
    return predicted_class

# Example usage
if __name__ == "__main__":
    test_image_path = "assets/images/sample-2.JPG"
    predict_image(test_image_path)
