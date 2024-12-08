# image-classifier-flask
Image Classification project from 0-9 number.

This project is a web application built using Flask and TensorFlow/Keras that allows users to upload an image, which is then classified using a pre-trained machine learning model. Here's a brief overview of how it works:

Project Overview
Model Creation (image_classification.py):

This file loads a pre-trained machine learning model (in this case, a Keras model) that has been trained to classify images into different classes (e.g., digits or objects).
The model is saved as a .h5 file, which is loaded and used to classify any new image uploaded via the web interface.
Web Interface (app.py):

This file contains the Flask application that handles the web server logic.
Users interact with the web application by visiting the webpage and uploading an image.
Once the image is uploaded, the app uses the machine learning model to classify the image and returns the prediction to the user.
Frontend (HTML Templates):

The HTML files (stored in the templates folder) provide the user interface for the application.
image_classification.html: This is the main page where users can upload an image for classification.
Flow of the Application
User Uploads an Image:

The user navigates to the homepage of the application where they can upload an image via a simple form (image_classification.html).
Image Processing:

The image is sent to the Flask backend (app.py).
In the backend, the image is saved in an "uploads" folder, then passed through the pre-trained model for classification. This is done in the image_classification.py file, which preprocesses the image (resizing, normalizing) and makes predictions.
Display Results:

Once the image is classified, the result (predicted class) is displayed on a new page (result.html).
The prediction is based on the output from the model, and the predicted label is displayed to the user.
Technical Components
Flask: Flask is used to handle the web server, routing, and serve the HTML templates.

Flask routes handle different URL paths (e.g., /, /upload).
The render_template() function is used to render HTML templates.
TensorFlow/Keras: TensorFlow is used to load the pre-trained model, and Keras is used for the high-level model interface.

The model predicts the class of the uploaded image, and the result is displayed on the webpage.
HTML Templates:

image_classification.html: Contains a form for image upload.
result.html: Displays the predicted class and the uploaded image.
Uploads Folder:

Images are saved in an "uploads" folder on the server for processing.
Workflow Summary
The user navigates to the homepage (/), uploads an image.
Flask handles the request, saves the image, and uses the model to classify it.
The result is displayed on a new page showing the predicted class.
Key Features:
Image Classification: Uses a machine learning model to classify images.
Flask Web Application: Simple, lightweight web interface for user interaction.
Real-time Prediction: The user sees the classification result immediately after uploading the image.
This project can be easily expanded to handle more complex datasets, additional model architectures, or more interactive front-end features. It is a great example of combining machine learning with web development to build an AI-powered application.
