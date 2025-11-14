import os
import pickle
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy import ndimage
from scipy.fft import fft2, fftshift

# ---------------------------
# Paths (local)
# ---------------------------
FLATFIELD_RESIDUALS_PATH = "Datasets/flatfield_residuals.pkl"
FP_OUT_PATH = "models/scanner_fingerprints.pkl"
ORDER_NPY = "models/fp_keys.npy"

RES_PATH = "Datasets/official_wiki_residuals.pkl"
FEATURES_OUT = "models/features.pkl"
ENHANCED_OUT = "models/enhanced_features.pkl"

# ---------------------------
# 1) Compute scanner fingerprints
# ---------------------------
with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
    flatfield_residuals = pickle.load(f)

scanner_fingerprints = {}
print("🔄 Computing fingerprints from Flatfields...")
for scanner, residuals in flatfield_residuals.items():
    if not residuals:
        continue
    stack = np.stack(residuals, axis=0)       # (num_images, H, W)
    fingerprint = np.mean(stack, axis=0)      # average residual
    scanner_fingerprints[scanner] = fingerprint

# Save fingerprints
with open(FP_OUT_PATH, "wb") as f:
    pickle.dump(scanner_fingerprints, f)

# Save deterministic scanner order
fp_keys = sorted(scanner_fingerprints.keys())
np.save(ORDER_NPY, np.array(fp_keys))
print(f"✅ Saved {len(scanner_fingerprints)} fingerprints and fp_keys.npy")

# ---------------------------
# 2) PRNU Features (cross-correlation)
# ---------------------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

# Load residuals
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

# Build features
features, labels = [], []
for dataset_name in ["Official", "Wikipedia"]:
    print(f"🔄 Computing PRNU features for {dataset_name} ...")
    for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                vec = [corr2d(res, scanner_fingerprints[k]) for k in fp_keys]
                features.append(vec)
                labels.append(scanner)

# Save features
with open(FEATURES_OUT, "wb") as f:
    pickle.dump({"features": features, "labels": labels}, f)
print(f"✅ Saved features shape: {len(features)} x {len(features[0])}")

# ---------------------------
# 3) Enhanced Features (FFT + LBP + Texture)
# ---------------------------
def extract_enhanced_features(residual):
    """FFT + LBP + statistical texture"""
    # FFT Features
    fft_img = np.abs(fft2(residual))
    fft_img = fftshift(fft_img)
    h, w = fft_img.shape
    center_h, center_w = h//2, w//2
    low_freq = np.mean(fft_img[center_h-20:center_h+20, center_w-20:center_w+20])
    mid_freq = np.mean(fft_img[center_h-60:center_h+60, center_w-60:center_w+60]) - low_freq
    high_freq = np.mean(fft_img) - low_freq - mid_freq

    # LBP
    lbp = local_binary_pattern(residual, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0,25), density=True)

    # Gradient / texture
    grad_x = ndimage.sobel(residual, axis=1)
    grad_y = ndimage.sobel(residual, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    texture_features = [
        np.std(residual),
        np.mean(np.abs(residual)),
        np.std(grad_mag),
        np.mean(grad_mag)
    ]

    return [low_freq, mid_freq, high_freq] + lbp_hist.tolist() + texture_features

# Extract enhanced features
enhanced_features, enhanced_labels = [], []
for dataset_name in ["Official", "Wikipedia"]:
    print(f"🔄 Extracting enhanced features for {dataset_name} ...")
    for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                feat = extract_enhanced_features(res)
                enhanced_features.append(feat)
                enhanced_labels.append(scanner)

# Save enhanced features
with open(ENHANCED_OUT, "wb") as f:
    pickle.dump({"features": enhanced_features, "labels": enhanced_labels}, f)
print(f"✅ Enhanced features shape: {len(enhanced_features)} x {len(enhanced_features[0])}")
print(f"✅ Saved enhanced features to {ENHANCED_OUT}")
