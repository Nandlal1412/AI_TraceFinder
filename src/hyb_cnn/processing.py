"""
processing.py
Data loading and preprocessing functions.
- Handles `Flatfield`, `Official`, `Wikipedia` datasets.
- Produces: residual images (256x256x1) and saves them as pickle files.
"""

import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: wavelet denoising
import pywt
from skimage.restoration import denoise_wavelet
from scipy.signal import wiener as scipy_wiener

# ---------------------------
# 1) Global Parameters
# ---------------------------
IMG_SIZE = (256, 256)
DENOISE_METHOD = "wavelet"  # "wavelet" or "wiener"
MAX_WORKERS = 8  # for ThreadPoolExecutor

# ---------------------------
# 2) Helper Functions
# ---------------------------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def denoise_wavelet_img(img):
    """Wavelet denoising."""
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

def preprocess_image(fpath, method=DENOISE_METHOD):
    """Common preprocessing: read, gray, resize, normalize, denoise, return residual."""
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = to_gray(img)
    img = resize_to(img)
    img = normalize_img(img)
    # Denoising
    if method == "wiener":
        den = scipy_wiener(img, mysize=(5,5))
    elif method == "wavelet":
        den = denoise_wavelet_img(img)
    else:
        raise ValueError(f"Unknown denoise method: {method}")
    return (img - den).astype(np.float32)

def process_folder(base_dir, use_dpi_subfolders=True):
    """
    Process all images under a folder.
    If use_dpi_subfolders=True, expects folder structure: scanner/dpi/*.tif
    Returns nested dict: residuals[scanner][dpi] = list_of_residuals
    """
    residuals_dict = {}
    scanners = sorted(os.listdir(base_dir))

    for scanner in tqdm(scanners, desc="Scanners"):
        scanner_dir = os.path.join(base_dir, scanner)
        if not os.path.isdir(scanner_dir):
            continue

        residuals_dict[scanner] = {}

        if use_dpi_subfolders:
            for dpi in os.listdir(scanner_dir):
                dpi_dir = os.path.join(scanner_dir, dpi)
                if not os.path.isdir(dpi_dir):
                    continue
                files = [os.path.join(dpi_dir, f) for f in os.listdir(dpi_dir)
                         if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
                residuals_dict[scanner][dpi] = parallel_process_images(files)
        else:
            # Flatfield dataset: no dpi subfolders
            files = [os.path.join(scanner_dir, f) for f in os.listdir(scanner_dir)
                     if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
            residuals_dict[scanner] = parallel_process_images(files)

    return residuals_dict

def parallel_process_images(file_list):
    """Process images in parallel and return residuals list."""
    residuals = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(preprocess_image, f) for f in file_list]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                residuals.append(res)
    return residuals

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved residuals to {path}")

# ---------------------------
# 3) Main Execution
# ---------------------------
if __name__ == "__main__":
    BASE_DIR = "Datasets"

    # Official + Wikipedia (dpi subfolders)
    datasets = ["Official", "Wikipedia"]
    official_wiki_residuals = {}

    for dataset in datasets:
        print(f"\n🔄 Processing {dataset} dataset...")
        dataset_dir = os.path.join(BASE_DIR, dataset)
        official_wiki_residuals[dataset] = process_folder(dataset_dir, use_dpi_subfolders=True)

    OUT_PATH = os.path.join(BASE_DIR, "official_wiki_residuals.pkl")
    save_pickle(official_wiki_residuals, OUT_PATH)

    # Flatfield dataset (no dpi subfolders)
    print("\n🔄 Processing Flatfield dataset...")
    flatfield_dir = os.path.join(BASE_DIR, "Flatfield")
    flatfield_residuals = process_folder(flatfield_dir, use_dpi_subfolders=False)
    OUT_PATH_FLAT = os.path.join(BASE_DIR, "flatfield_residuals.pkl")
    save_pickle(flatfield_residuals, OUT_PATH_FLAT)

    # Summary
    total_scanners = len(flatfield_residuals)
    total_images = sum(len(v) for v in flatfield_residuals.values())
    print(f"\n✅ Done. Flatfield: {total_scanners} scanners, {total_images} images")
