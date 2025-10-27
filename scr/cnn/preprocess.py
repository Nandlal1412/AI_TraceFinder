import tifffile as tiff
import numpy as np
import cv2

def load_and_preprocess(img_path, size=(256,256), gray=False):
    """Load TIFF (or other images), resize, normalize"""
    try:
        img = tiff.imread(img_path) 
        
        if img.ndim == 2:
            if gray:
                img = np.expand_dims(img, axis=-1)
            else:
                img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:  
            img = img[:, :, :3]

        
        img = cv2.resize(img, size)
        img = img.astype("float32") / 255.0
        return img
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")
        raise
