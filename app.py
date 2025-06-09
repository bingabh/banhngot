import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Danh sách nhãn tương ứng với output
class_names = ['donut', 'su kem', 'sừng bò', 'tart trứng']

# Load model
def load_mobilenet_model():
    return load_model('MobileNet_RGB-2506.h5')

model = load_mobilenet_model()

# Cấu hình đầu vào ảnh
IMG_SIZE = (150, 150)

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Giao diện
st.title("📷 MobileNet Image Classifier Demo")
st.write("Upload một ảnh để mô hình dự đoán.")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã upload", use_container_width=True)

    with st.spinner('⏳ Đang dự đoán...'):
        input_data = preprocess_image(img)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        class_name = class_names[predicted_class]
        st.success(f"✅ Kết quả dự đoán: **{class_name}**")