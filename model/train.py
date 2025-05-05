"""
model/train.py
Placeholder for model training script.
This script will handle loading MRI data, preprocessing, training the CNN, and saving the trained model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from glob import glob

# Configuration
PROCESSED_DATA_DIR = 'data/processed/'
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

# Load data
def load_data():
    x = []
    y = []
    for file in glob(os.path.join(PROCESSED_DATA_DIR, '*.npy')):
        img = np.load(file)
        x.append(img)
        # Simple label extraction: expects filenames like 'yes_1.npy' (tumor) or 'no_2.npy' (no tumor)
        label = 1 if 'yes' in file.lower() else 0
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    x = x[..., np.newaxis]  # Add channel dimension
    return x, y

# Define CNN model
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading data...")
    x, y = load_data()
    print(f"Data shape: {x.shape}, Labels shape: {y.shape}")
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 1))
    print("Training model...")
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
    model.save('model/brain_tumor_cnn.h5')
    print("Model training complete and saved as model/brain_tumor_cnn.h5")