import os
import glob
import pickle
import numpy as np
import tensorflow as tf
import pywt
import cv2
from skimage.feature import local_binary_pattern
import csv

# -----------------------
# Paths
# -----------------------
ART_DIR = "models"  
FP_PATH = os.path.join(ART_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(ART_DIR, "fp_keys.npy")
CKPT_PATH = os.path.join(ART_DIR, "scanner_hybrid.keras")
IMG_SIZE = (256, 256)

# -----------------------
# Load model & preprocessors
# -----------------------
hyb_model = tf.keras.models.load_model(CKPT_PATH, compile=False)

with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "rb") as f:
    le_inf = pickle.load(f)

with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "rb") as f:
    scaler_inf = pickle.load(f)

with open(FP_PATH, "rb") as f:
    scanner_fps_inf = pickle.load(f)

fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()

# -----------------------
# Utility functions
# -----------------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    return [float(mag[(r >= bins[i]) & (r < bins[i+1])].mean() if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32).tolist()

def preprocess_residual_pywt(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    return (img - den).astype(np.float32)

def make_feats_from_res(res):
    v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
    v_fft  = fft_radial_energy(res)
    v_lbp  = lbp_hist_safe(res)
    v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
    return scaler_inf.transform(v)

def predict_scanner_hybrid(image_path):
    res = preprocess_residual_pywt(image_path)
    x_img = np.expand_dims(res, axis=(0,-1))
    x_feat = make_feats_from_res(res)
    prob = hyb_model.predict([x_img, x_feat], verbose=0)
    idx = int(np.argmax(prob))
    label = le_inf.classes_[idx]
    conf = float(prob[0, idx]*100)
    return label, conf

# -----------------------
# Predict all images in a folder and save CSV
# -----------------------
def predict_folder(folder_path, output_csv="results.csv", exts=("*.tif","*.png","*.jpg","*.jpeg")):
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    print(f"Found {len(image_files)} images in {folder_path}")

    results = []
    for img_path in image_files:
        try:
            label, conf = predict_scanner_hybrid(img_path)
            results.append((img_path, label, conf))
            print(f"{img_path} -> {label} | {conf:.2f}%")
        except Exception as e:
            print(f"⚠️ Error {img_path}: {e}")

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Predicted_Label", "Confidence(%)"])
        writer.writerows(results)
    print(f"\n✅ Predictions saved to {output_csv}")
    return results

# -----------------------
# Run script
# -----------------------
if __name__ == "__main__":
    folder_to_test = "Test" 
    predict_folder(folder_to_test, output_csv="hybrid_folder_results.csv")
