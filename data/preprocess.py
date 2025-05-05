"""
data/preprocess.py
Placeholder for data preprocessing script.
This script will handle loading raw MRI images, preprocessing (resizing, normalization, augmentation), and saving processed data for model training.
"""

import os
import cv2
import numpy as np
from glob import glob

# Configuration
RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
IMG_SIZE = (128, 128)

def ensure_dirs():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0,1]
    return img

def preprocess_dataset():
    ensure_dirs()
    image_paths = glob(os.path.join(RAW_DATA_DIR, '*.jpg')) + glob(os.path.join(RAW_DATA_DIR, '*.png'))
    print(f"Found {len(image_paths)} images for preprocessing.")
    for img_path in image_paths:
        img = preprocess_image(img_path)
        filename = os.path.basename(img_path)
        np.save(os.path.join(PROCESSED_DATA_DIR, filename.replace('.jpg', '.npy').replace('.png', '.npy')), img)
    print("Preprocessing complete. Processed images saved as .npy files.")

if __name__ == "__main__":
    preprocess_dataset()
