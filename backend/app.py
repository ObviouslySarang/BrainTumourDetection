from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('model', 'brain_tumor_cnn.h5')
IMG_SIZE = (128, 128)

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    return 'Brain Tumor Detection API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = img[np.newaxis, ..., np.newaxis]  # Shape: (1, 128, 128, 1)
    pred = model.predict(img)[0][0]
    confidence = float(pred) if pred > 0.5 else 1 - float(pred)
    result = 'Tumor' if pred > 0.5 else 'No Tumor'
    return jsonify({'result': result, 'confidence': round(confidence * 100, 2)})

if __name__ == '__main__':
    app.run(debug=True)
