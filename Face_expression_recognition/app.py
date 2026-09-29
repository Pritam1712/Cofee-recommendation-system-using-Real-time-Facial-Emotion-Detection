from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import cv2
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('facial_emotion_model.h5')

# Define emotion-to-coffee mapping with detailed coffee attributes
coffee_data = {
    "Espresso": [5, 0, 1, 0, 63, 0],
    "Americano": [10, 0, 2, 0.5, 77, 0],
    "Ristretto": [5, 0, 1, 0, 65, 0],
    "Lungo": [10, 0, 2, 0.5, 75, 0],
    "Cappuccino": [80, 3, 9, 4, 80, 10],
    "Latte": [150, 6, 15, 8, 90, 20],
    "Flat White": [120, 5, 12, 6, 85, 15],
    "Macchiato": [15, 0.5, 2, 1, 75, 5],
    "Mocha": [180, 8, 25, 5, 95, 25],
    "Affogato": [200, 10, 30, 6, 100, 30],
    "Iced Coffee": [5, 0, 1, 0, 80, 0],
    "Iced Latte": [140, 5, 18, 7, 90, 15],
    "Cold Brew": [5, 0, 1, 0, 200, 0],
    "Nitro Coffee": [10, 0.5, 2, 1, 215, 2]
}

emotion_to_coffee = {
    'Angry': ['Espresso', 'Ristretto', 'Macchiato', 'Iced Coffee', 'Cold Brew', 'Nitro Coffee'],
    'Happy': ['Espresso', 'Americano', 'Lungo', 'Cappuccino', 'Latte', 'Flat White', 'Macchiato', 
              'Mocha', 'Affogato', 'Iced Coffee', 'Iced Latte', 'Cold Brew', 'Nitro Coffee'],
    'Neutral': ['Americano', 'Lungo', 'Cappuccino', 'Latte', 'Flat White', 'Mocha', 'Iced Latte', 'Cold Brew'],
    'Sad': ['Latte', 'Flat White', 'Mocha', 'Affogato', 'Iced Latte']
}

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route: Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Route: Detect Emotion and Recommend Coffee
@app.route("/detect", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(file_path)

            # Preprocess image & predict
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            predicted_class = list(emotion_to_coffee.keys())[np.argmax(prediction)]  # Get highest probability class

            # Get coffee recommendations
            recommended_coffee = emotion_to_coffee.get(predicted_class, [])

            # Get coffee details
            coffee_details = {coffee: coffee_data[coffee] for coffee in recommended_coffee}

            # Save prediction overlay
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Predicted Emotion: {predicted_class}")
            plt.savefig(file_path)  # Save the image with title

            return render_template("index.html", prediction=predicted_class, image_file=file_path, coffee_details=coffee_details)

    return render_template("index.html", prediction=None, image_file=None, coffee_details=None)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
