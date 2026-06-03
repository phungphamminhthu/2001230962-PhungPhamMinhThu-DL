
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model(
    "logo_classification_model.keras"
)

# Danh sách class
class_names = [
    "Accessories",
    "Clothes",
    "Cosmetic",
    "Electronic",
    "Food",
    "Institution",
    "Leisure",
    "Medical",
    "Necessities",
    "Transportation"
]

# =========================
# Cấu hình trang
# =========================
st.set_page_config(
    page_title="Logo Classification",
    page_icon="",
    layout="centered"
)

st.title(" Logo Classification")
st.write("Tải lên ảnh logo để hệ thống nhận diện loại logo")

# =========================
# Upload ảnh
# =========================
uploaded_file = st.file_uploader(
    "Chọn ảnh logo",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Dự đoán
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Ảnh đã tải lên",
        use_container_width=True
    )

    # Resize đúng kích thước model
    img = image.resize((128, 128))

    # Chuyển sang numpy
    img = np.array(img)

    # Chuẩn hóa
    img = img / 255.0

    # Thêm chiều batch
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_label = class_names[predicted_class]

    st.success(
        f"Loại logo dự đoán: {predicted_label}"
    )

    st.info(
        f"Độ tin cậy: {confidence * 100:.2f}%"
    )

    # =========================
    # Top 3 kết quả
    # =========================
    st.subheader(" Top 3 dự đoán")

    top3_idx = np.argsort(prediction[0])[-3:][::-1]

    for idx in top3_idx:
        st.write(
            f"**{class_names[idx]}** : {prediction[0][idx] * 100:.2f}%"
        )

    # =========================
    # Thanh xác suất
    # =========================
    st.subheader(" Xác suất các lớp")

    for i, class_name in enumerate(class_names):
        prob = float(prediction[0][i])

        st.write(
            f"{class_name}: {prob * 100:.2f}%"
        )

        st.progress(prob)