# ui.py
import streamlit as st
from predictor import predict_image

st.title("🍅 Tomato Disease Detector")
st.write("Upload an image to get predictions + remedies")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    st.image(file, caption="Uploaded Image", width=300)
    preds = predict_image(file.read())

    st.subheader("🔍 Top Predictions")
    for p in preds:
        st.write(f"**{p['class']}** — Confidence: `{p['confidence']:.4f}`")
        st.info(p["remedy"])