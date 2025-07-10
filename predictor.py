# predictor.py
from utils import preprocess_image
import joblib
import numpy as np

def predict_image(image_path: str) -> str:
    model = joblib.load("svm_model.pkl")
    features = preprocess_image(image_path).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        max_prob = max(probs)
        prediction = model.predict(features)[0]

        if max_prob < 0.85:  # stricter threshold
            return "Unknown ❓ (Low confidence)"

        return "Dog 🐶" if prediction == 1 else "Cat 🐱"
    else:
        prediction = model.predict(features)[0]
        return "Dog 🐶" if prediction == 1 else "Cat 🐱"
