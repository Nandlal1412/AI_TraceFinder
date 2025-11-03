import numpy as np
import tensorflow as tf
import cv2
import os
from feature_extraction import extract_noise
from preprocess import load_and_preprocess

# Paths
MODEL_PATH = r"models\dual_branch_cnn.h5"
LABELS = sorted(os.listdir(r"Datasets\Official"))  # assumes Official\<device_folder>

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_single(image_path):
    # Preprocess official image (RGB)
    img = load_and_preprocess(image_path, size=(256, 256), gray=False)
    img = np.expand_dims(img, axis=0)  # shape: (1, 256, 256, 3)

    # Extract noise map
    noise = extract_noise(image_path, size=(256, 256))

    # Ensure noise is grayscale
    if noise.ndim == 3 and noise.shape[-1] == 3:
        noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)

    noise = np.expand_dims(noise, axis=-1)  # shape: (256, 256, 1)
    noise = np.expand_dims(noise, axis=0)   # shape: (1, 256, 256, 1)

    # Optional: Debug input shapes
    print("Official input shape:", img.shape)
    print("Noise input shape:", noise.shape)

    # Predict
    preds = model.predict({"official_input": img, "noise_input": noise})
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    return LABELS[pred_class], confidence

if __name__ == "__main__":
    test_image = r"Datasets\Official\Canon9000-1\300\s4_73.tif"  # example path

    if not os.path.exists(test_image):
        print("❌ File not found:", test_image)
    else:
        device, conf = predict_single(test_image)
        print(f"✅ Predicted Device: {device} (Confidence: {conf:.2f})")