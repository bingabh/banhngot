import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Danh sách nhãn tương ứng với output
class_names = ['donut', 'su kem', 'sừng bò', 'tart trứng']

# ✅ Load model chỉ 1 lần
@st.cache_resource
def load_mobilenet_model():
    return load_model('MobileNet_RGB-2506.h5')

model = load_mobilenet_model()

# Cấu hình ảnh đầu vào
IMG_SIZE = (150, 150)

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def show_prediction_chart(preds, labels):
    fig, ax = plt.subplots()
    bars = ax.barh(labels, preds[0], color="#f63366")
    ax.set_xlim(0, 1)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f"{preds[0][i]*100:.1f}%", va='center')
    st.pyplot(fig)

# Title
st.title("🍰 Phân Loại Bánh Ngọt với MobileNet")
st.write("Tải ảnh lên để dự đoán loại bánh bằng mô hình học sâu.")

# Upload
st.markdown("### 📥 Chọn ảnh:")
uploaded_file = st.file_uploader("Tải ảnh bánh (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Xử lý ảnh
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img, caption="📷 Ảnh đã tải lên", use_container_width=True)

    with st.spinner("⏳ Đang dự đoán..."):
        input_data = preprocess_image(img)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        class_name = class_names[predicted_class]

    with col2:
        st.markdown("### ✅ Kết quả dự đoán:")
        st.success(f"**{class_name}**")

    st.markdown("---")
    st.markdown("### 📊 Biểu đồ xác suất dự đoán:")
    show_prediction_chart(prediction, class_names)

else:
    st.info("📂 Vui lòng chọn ảnh để bắt đầu dự đoán.")
    st.image("https://i.imgur.com/IDxkq8F.jpeg", caption="Ví dụ: Ảnh bánh ngọt", use_container_width=True)
