# svm_model.py
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from utils import preprocess_image

print("📦 Starting model training...")

def load_data(data_dir="dataset/train", size=(64, 64)):
    X, y = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            label = 1 if "dog" in file.lower() else 0
            path = os.path.join(data_dir, file)
            features = preprocess_image(path, size)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()

print(f"📊 Loaded {len(X)} images.")

if len(X) == 0:
    print("❌ No images found in dataset/train/. Exiting.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

joblib.dump(model, "svm_model.pkl")
print("💾 Model saved as 'svm_model.pkl'")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc:.2f}")
