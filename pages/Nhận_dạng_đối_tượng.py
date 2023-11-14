import cv2
import streamlit as st
import numpy as np

st.set_page_config(page_title="Nhận dạng đối tượng YOLOv4", page_icon="🌐")
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-042.jpg");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Chọn một tệp tin ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh từ tệp tin
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with open('./nd/coco_class_names.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet('./nd/yolov4.cfg', './nd/yolov4.weights')

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    for (classId, score, box) in zip(classIds, scores, boxes):
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2)

    # Chuyển đổi ảnh OpenCV thành định dạng RGB để hiển thị trong Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh và kết quả nhận dạng đối tượng sử dụng Streamlit
    st.image(img_rgb, channels="RGB")

    for (classId, score, box) in zip(classIds, scores, boxes):
        st.write(f"Class: {classes[classId]}, Score: {score}")