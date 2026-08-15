import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. عنوان الصفحة والشكل العام
st.set_page_config(page_title="Pneumonia Detection CAD System")
st.title(" Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to get real-time classification and confidence score.")

# 2. تحميل الموديل المحفوظ
@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model('pneumonia_detection_model.keras')

model = load_pneumonia_model()
class_names = ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRAL']

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray Image', use_container_width=True)
    st.write("")
    
    img = image.convert('L') # Grayscale
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1)) #  (1, 256, 256, 1)

    if st.button('Diagnose Image'):
        with st.spinner('Analyzing X-Ray...'):
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence Score:** {confidence:.2f}%")