# utils.py
import cv2
import numpy as np

def preprocess_image(img_path: str, size=(64, 64)) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.flatten() / 255.0  # Normalize
