import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import io
import pandas as pd

# ⚙️ Cấu hình trang
st.set_page_config(page_title="Phân Loại Bánh Ngọt", layout="wide")

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Danh sách nhãn tương ứng với output và cấu hình ảnh đầu vào
class_names = ['donut', 'su kem', 'sừng bò', 'tart trứng']
IMG_SIZE = (150, 150)

# ✅ Load model chỉ 1 lần
@st.cache_resource
def load_mobilenet_model():
    return load_model('MobileNet_RGB-2506.h5')

model = load_mobilenet_model()

# 📦 Hàm tiền xử lý ảnh
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# 📊 Vẽ biểu đồ xác suất dự đoán
def show_prediction_chart(preds, labels):
    fig, ax = plt.subplots()
    bars = ax.barh(labels, preds[0], color="#f63366")
    ax.set_xlim(0, 1)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f"{preds[0][i]*100:.1f}%", va='center')
    st.pyplot(fig)

# 🏷️ Tiêu đề với thiết kế đẹp
st.markdown("""
    <div style='background: linear-gradient(to right, #6a11cb, #2575fc); padding: 1.5rem; border-radius: 10px; text-align: center;'>
        <h1 style='color: white;'>🍰 Phân Loại Bánh Ngọt với MobileNet</h1>
        <p style='color: white;'>Tải ảnh lên và để AI phân tích loại bánh với độ chính xác cao</p>
    </div>
""", unsafe_allow_html=True)

# 🔁 Khởi tạo biến phiên
if "history" not in st.session_state:
    st.session_state.history = []

result_text = None
image_bytes = None
prediction = None
img = None
class_name = ""
confidence = 0.0

# 📐 Bố cục 3 cột
left_col, center_col, right_col = st.columns([1.2, 1.8, 1.2])

# ⬅️ Lịch sử bên trái
with left_col:
    st.markdown("### 📝 Lịch sử phân tích")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(f"**{idx}.** `{item['filename']}` → **{item['class']}** ({item['confidence']})")
    else:
        st.info("Chưa có lịch sử phân tích.")

# 🔍 Trung tâm: tải và phân tích ảnh
with center_col:
    st.markdown("### 📥 Tải ảnh bánh (JPG, PNG):")
    uploaded_file = st.file_uploader("Chọn một ảnh bánh để phân tích", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Ảnh đã tải lên", use_container_width=True)

        if st.button("🔍 Phân tích ảnh"):
            with st.spinner("⏳ Đang dự đoán..."):
                input_data = preprocess_image(img)
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
                class_name = class_names[predicted_class]

            result_container = st.container()
            with result_container:
                if confidence >= 0.98:
                    st.markdown(f"""
                        <div style='background: linear-gradient(to right, #6a11cb, #2575fc); padding: 1.5rem; border-radius: 10px; text-align: center;'>
                            <h2 style='color: white;'>🍩 <strong>{class_name}</strong> với độ tin cậy {confidence*100:.2f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    result_text = f"Dự đoán: {class_name}\nĐộ tin cậy: {confidence*100:.2f}%"
                    st.session_state.history.append({
                        "filename": uploaded_file.name,
                        "class": class_name,
                        "confidence": f"{confidence*100:.2f}%"
                    })
                else:
                    st.markdown(f"""
                        <div style='background: linear-gradient(to right, #6a11cb, #2575fc); padding: 1.5rem; border-radius: 10px; text-align: center;'>
                            <h2 style='color: white;'>🚫 Không thể nhận diện ảnh này với độ tin cậy đủ cao</h2>
                            <p style='color: white;'>Kết quả: {class_name} ({confidence*100:.2f}%)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    result_text = f"Không thể nhận diện ảnh với độ tin cậy đủ cao.\nDự đoán: {class_name} ({confidence*100:.2f}%)"
                    st.session_state.history.append({
                        "filename": uploaded_file.name,
                        "class": "Không xác định",
                        "confidence": f"{confidence*100:.2f}%"
                    })

            image_buffer = io.BytesIO()
            img.save(image_buffer, format='PNG')
            image_bytes = image_buffer.getvalue()

            st.download_button(
                label="📄 Tải kết quả văn bản",
                data=result_text,
                file_name="ketqua_du_doan.txt",
                mime="text/plain"
            )

            st.download_button(
                label="🖼️ Tải ảnh đã phân tích",
                data=image_bytes,
                file_name="anh_du_doan.png",
                mime="image/png"
            )
    else:
        st.info("📂 Vui lòng chọn ảnh để bắt đầu.")
        st.image("https://i.imgur.com/IDxkq8F.jpeg", caption="Ví dụ: Ảnh bánh ngọt", use_container_width=True)

# ➡️ Biểu đồ bên phải
with right_col:
    st.markdown("### 📊 Biểu đồ xác suất")
    if uploaded_file and prediction is not None:
        show_prediction_chart(prediction, class_names)

        # Bảng dữ liệu xác suất
        probs_df = pd.DataFrame({
            "Loại bánh": class_names,
            "Xác suất (%)": [f"{p*100:.2f}%" for p in prediction[0]]
        })
        st.markdown("#### 📋 Bảng xác suất")
        st.dataframe(probs_df, use_container_width=True)

# 📎 Footer đẹp
st.markdown("---")
st.markdown("""
    <div style='
        text-align: center;
        padding: 20px;
        font-size: 18px;
        color: #31333F;
    '>
        🚀 Phát triển bởi <strong>Nhóm 14</strong> · Sử dụng Python · TensorFlow · Streamlit
    </div>
""", unsafe_allow_html=True)