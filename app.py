import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Auto-Inspect AI", page_icon="🚗")

st.title("Automotive Defect Detection")
st.markdown("Upload a product image to check if it is DEFECTIVE or OK")

# =====================
# LOAD MODEL (FIXED)
# =====================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('defect_model.keras')

model = load_my_model()

# =====================
# UPLOAD IMAGE
# =====================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Analyzing image...")

    # Preprocess
    img = image.resize((300, 300))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    # =====================
    # RESULT
    # =====================
    if prediction < 0.5:
        confidence = (1 - prediction) * 100
        st.error(f"❌ DEFECT DETECTED ({confidence:.2f}% confidence)")
        st.write("This part should be rejected from production line.")
    else:
        confidence = prediction * 100
        st.success(f"✅ QUALITY OK ({confidence:.2f}% confidence)")
        st.write("This part meets quality standards.")
