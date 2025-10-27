import cv2
import numpy as np
from preprocess import load_and_preprocess

def extract_noise(img_path, size=(256,256)):
    """Extract scanner noise (flatfield-like)"""
    img = load_and_preprocess(img_path, size=size, gray=True)
    
    denoised = cv2.GaussianBlur(img, (5,5), 0)
    noise = img - denoised
    return noise

