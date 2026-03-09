import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# Optional CSS for enhanced theme
def add_custom_css():
    st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stFileUploader, .stImage {
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .caption-box {
        padding: 12px;
        background-color: #222831;
        border-left: 5px solid #4CAF50;
        border-radius: 5px;
        font-size: 18px;
        color: #eeeeee;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("organic_vs_recycle_model.h5")

# Preprocess image for prediction
def preprocess_image(image, target_size=(128, 128)):
    image = ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype('float32') / 255.0
    if img_array.shape[-1] == 4:  # Remove alpha channel if present
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function using threshold 0.5
def predict_class(model, image_array):
    prediction = model.predict(image_array)
    confidence = float(prediction[0][0])
    label = "Organic" if confidence < 0.5 else "Recyclable"
    confidence_display = 1 - confidence if label == "Organic" else confidence
    return label, confidence_display

# Streamlit UI
def main():
    add_custom_css()
    st.title("Eco-Sort : Organic vs Recyclable")
    st.markdown("Upload an image to find out if it's **Organic** or **Recyclable**.")

    image_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Loading model..."):
            model = load_model()

        image_array = preprocess_image(image)

        if st.button("Classify"):
            with st.spinner("Classifying..."):
                label, confidence = predict_class(model, image_array)
                st.markdown(
                    f'<div class="caption-box">'
                    f'🧪 <strong>Prediction:</strong> {label}<br>'
                    f'📊 <strong>Confidence:</strong> {confidence:.2f}'
                    f'</div>',
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()

st.markdown(
    """
    <style>
    .credit {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: #999999;
    }
    .credit a {
        color: #999999;
        text-decoration: none;
    }
    .credit a:hover {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)
