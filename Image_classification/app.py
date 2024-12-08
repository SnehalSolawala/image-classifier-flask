import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model (example: MNIST model)
model = tf.keras.models.load_model('mnist_classifier.h5')

# Path to save uploaded images (optional, but useful for testing)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Route to handle the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the main page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        img = Image.open(image_file.stream).convert('L')  # Convert to grayscale (if needed)

        # Preprocess the image to match the model's expected input
        img = img.resize((28, 28))  # Resize to 28x28 for MNIST model
        img_array = np.array(img) / 255.0  # Normalize the image

        # Expand dimensions to match the input shape (batch size of 1)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Convert numpy int64 to a native Python int
        predicted_class = int(predicted_class)

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
