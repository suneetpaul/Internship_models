# app.py - CORRECTED for 32x32 Images (matches actual model input shape)

import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Page config + white theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Digit Classification", page_icon="🔢", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #1A1A1A;
    }
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #1A1A1A;
    }
    div[data-testid="stMetric"] {
        background-color: #F7F7F9;
        border: 1px solid #E6E6E6;
        border-radius: 10px;
        padding: 12px;
    }
    .stButton > button {
        background-color: #2563EB;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F7F7F9;
        border: 1px dashed #C9C9C9;
    }
    .stDataFrame {
        background-color: #FFFFFF;
    }
    hr {
        border-top: 1px solid #E6E6E6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Digit Classification (0-9)")
st.markdown("---")

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_model (1).keras")

# This model expects (32, 32, 1) inputs - confirmed from model.input_shape
MODEL_INPUT_SIZE = (32, 32)


@st.cache_resource
def load_cnn_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Model not found or failed to load: {e}")
        return None


CLASS_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
model = load_cnn_model()

if model is None:
    st.stop()

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader("Choose an image (0-9)", type=["jpg", "jpeg", "png"])

    invert_colors = st.checkbox(
        "Invert colors (use if your digit is dark on a light background)",
        value=False,
        help="MNIST-style training data is usually a white digit on a black background. "
             "If your photo is black ink on white paper, turn this on.",
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🎯 Results")

    if image is not None:
        classify_clicked = st.button("🚀 Classify", use_container_width=True)

        if classify_clicked:
            with st.spinner("Processing..."):
                try:
                    # Convert to grayscale
                    img_grayscale = image.convert('L')

                    # Optionally invert (dark digit on light background -> light on dark)
                    if invert_colors:
                        img_grayscale = ImageOps.invert(img_grayscale)

                    # Resize to the model's actual expected input size (32x32)
                    img_resized = img_grayscale.resize(MODEL_INPUT_SIZE)

                    # Convert to array and normalize
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0

                    # Show what the model actually sees, for debugging
                    with st.expander("👀 Preview of preprocessed input"):
                        st.image(img_resized, caption=f"Resized to {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}, grayscale", width=150)

                    # Reshape: (32, 32) -> (1, 32, 32, 1)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = np.expand_dims(img_array, axis=-1)

                    # Predict
                    predictions = model.predict(img_array, verbose=0)
                    predicted_idx = int(np.argmax(predictions[0]))
                    confidence = float(predictions[0][predicted_idx])

                    st.success("✅ Done!")
                    st.markdown(
                        f"<p style='font-size:80px; text-align:center; color:#2563EB;'>{CLASS_LABELS[predicted_idx]}</p>",
                        unsafe_allow_html=True,
                    )

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Predicted", CLASS_LABELS[predicted_idx])
                    with col_m2:
                        st.metric("Confidence", f"{confidence*100:.2f}%")

                    st.progress(confidence)

                    # Top 5
                    st.write("**Top 5 Predictions:**")
                    top_5 = np.argsort(predictions[0])[::-1][:5]
                    df = pd.DataFrame({
                        'Digit': [CLASS_LABELS[i] for i in top_5],
                        'Confidence (%)': [f"{predictions[0][i]*100:.2f}" for i in top_5]
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor('white')
                    ax.set_facecolor('white')
                    ax.bar(CLASS_LABELS, predictions[0] * 100, color='#2563EB')
                    ax.set_ylabel('Probability (%)')
                    ax.set_title('All Digit Probabilities')
                    st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Click Classify")
    else:
        st.info("👈 Upload an image to get started")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>MNIST Digit Classification | Streamlit + TensorFlow</p>",
    unsafe_allow_html=True,
)
