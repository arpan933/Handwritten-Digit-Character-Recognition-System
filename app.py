import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf

st.set_page_config(page_title="Handwritten Recognition System", page_icon="✍️", layout="centered")
st.title("Handwritten Digit & Character Recognition System")
st.write("Upload a clear image containing one handwritten digit or character. The system preprocesses the image and predicts its class.")

MODEL_PATH = "models/emnist_model.keras"
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.autocontrast(image)

    arr = np.array(image, dtype=np.float32)

    # Detect background polarity
    border = np.concatenate([
        arr[0, :], arr[-1, :],
        arr[:, 0], arr[:, -1]
    ])

    if np.median(border) > 127:
        arr = 255 - arr

    # Remove background noise
    arr = np.clip(arr - np.median(arr), 0, 255)

    if arr.max() > 0:
        arr = arr / arr.max() * 255

    arr[arr < 70] = 0

    image = Image.fromarray(arr.astype("uint8"))

    # Crop character
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resize character
    w, h = image.size
    scale = 20 / max(w, h)

    image = image.resize(
        (max(1, round(w * scale)),
         max(1, round(h * scale))),
        Image.Resampling.LANCZOS
    )

    # Center character
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))
    canvas = ImageOps.autocontrast(canvas)

    # Normalize and match EMNIST orientation
    arr = np.array(canvas, dtype=np.float32) / 255.0

    return arr.T.reshape(1, 28, 28, 1)

model = load_model()
if model is None:
    st.warning("The trained model is not installed yet. Follow README.md to train it first.")
    st.code("python train.py --epochs 5")
    st.stop()

uploaded = st.file_uploader("Choose a handwritten image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width="stretch")
    x = preprocess_image(image)
    with st.expander("View preprocessed 28 × 28 image"):
     st.image((x[0, :, :, 0].T * 255).astype("uint8"), width=280)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    st.subheader(f"Prediction: {LABELS[idx]}")
    st.metric("Model confidence", f"{float(probs[idx]) * 100:.2f}%")
    top = np.argsort(probs)[-5:][::-1]
    st.write("Top 5 predictions")
    for i in top:
        st.write(f"{LABELS[int(i)]}: {float(probs[i]) * 100:.2f}%")
else:
    st.info("Tip: use a single centered character with a plain background for the best result.")

st.caption("Academic demonstration project. Predictions are estimates and may be incorrect.")
