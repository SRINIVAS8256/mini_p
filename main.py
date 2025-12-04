from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model - handle Keras 3.x compatibility issue
import tensorflow as tf
import h5py
import json

# Set environment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# CRITICAL FIX: Patch Flatten layer BEFORE importing/loading anything
# This fixes the Keras 3.x bug where Flatten receives list instead of tensor
from tensorflow.keras.layers import Flatten

original_compute_output_spec = Flatten.compute_output_spec

def patched_compute_output_spec(self, inputs):
    """Fix for Keras 3.x: handle list inputs to Flatten"""
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    elif isinstance(inputs, list) and len(inputs) > 1:
        # Take first element if multiple
        inputs = inputs[0]
    return original_compute_output_spec(self, inputs)

Flatten.compute_output_spec = patched_compute_output_spec

print("Loading model from models/model.h5...")

# Now try loading the model
try:
    from tensorflow.keras.models import load_model
    model = load_model('models/model.h5', compile=False, safe_mode=False)
    print("✓ Model loaded successfully!")
except Exception as e:
    error_str = str(e)
    print(f"Error: {error_str[:200]}")
    raise Exception(f"Could not load model. This model may need to be retrained with Keras 3.x. Error: {error_str[:300]}")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
