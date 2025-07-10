# app.py
import streamlit as st
from PIL import Image
from predictor import predict_image
import os
from datetime import datetime

# Custom style
def inject_css():
    st.markdown("""
        <style>
            .title { text-align: center; color: #5f259f; font-size: 2.5rem; font-weight: bold; }
            .subtitle { text-align: center; color: #8e44ad; font-size: 1.2rem; margin-bottom: 25px; }
            .result-box { margin-top: 25px; background-color: #e0e0ff; border-radius: 10px;
                          padding: 15px; font-size: 1.4rem; text-align: center; font-weight: 600; }
            .save-success { color: green; font-size: 1rem; text-align: center; margin-top: 10px; }
            .footer { margin-top: 50px; font-size: 0.85rem; color: #999; text-align: center; }
        </style>
    """, unsafe_allow_html=True)

inject_css()

st.markdown("<div class='title'>🐾 Cat vs Dog Image Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>(with Unknown Class Handling)</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
temp_path = "temp.jpg"

if uploaded_file:
    image = Image.open(uploaded_file)
    image.save(temp_path)
    st.image(image, caption="🖼️ Uploaded Image Preview", use_container_width=True)

    if st.button("🔍 Predict"):
        label = predict_image(temp_path)

        st.markdown(f"<div class='result-box'>Prediction Result: <strong>{label}</strong></div>", unsafe_allow_html=True)

        # Save result
        result_dir = "predicted_results"
        os.makedirs(result_dir, exist_ok=True)
        filename = f"{label.lower().split()[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(result_dir, filename)
        image.save(save_path)

        st.markdown(f"<div class='save-success'>✅ Image saved as: <code>{filename}</code></div>", unsafe_allow_html=True)

        with open(save_path, "rb") as f:
            st.download_button("📥 Download Prediction", data=f, file_name=filename, mime="image/jpeg")

st.markdown("<div class='footer'>Made with 💜 using Streamlit</div>", unsafe_allow_html=True)
