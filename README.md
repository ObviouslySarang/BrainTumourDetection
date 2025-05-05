# Brain Tumor Detection in MRI Images using Deep CNN

This project provides a web-based tool for automatic brain tumor detection in MRI images using a deep convolutional neural network (CNN).

## Features
- Upload brain MRI images and get instant tumor detection results
- Modern, user-friendly frontend (HTML/CSS/JS)
- Flask backend with TensorFlow CNN model
- Easy-to-follow modular code structure

## Project Structure
```
requirements.txt
backend/
    app.py           # Flask API for inference
frontend/
    index.html       # Web UI for image upload and results
model/
    train.py         # Model training script
    brain_tumor_cnn.h5 (generated after training)
data/
    preprocess.py    # Data preprocessing script
    raw/             # Place raw MRI images here
    processed/       # Preprocessed .npy files
```

## Getting Started
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Preprocess data:**
   - Place MRI images in `data/raw/` (filenames with 'yes' for tumor, 'no' for no tumor)
   - Run: `python data/preprocess.py`
3. **Train the model:**
   ```sh
   python model/train.py
   ```
4. **Run the backend:**
   ```sh
   python backend/app.py
   ```
5. **Open the frontend:**
   - Open `frontend/index.html` in your browser

## Deployment
- **Frontend:** Deploy `frontend/index.html` using GitHub Pages, Netlify, or Vercel.
- **Backend:** Deploy Flask API on Render, PythonAnywhere, Railway, or Heroku.
- Update the API URL in `index.html` to your backend's public URL after deployment.

## License
For research and educational use only.
