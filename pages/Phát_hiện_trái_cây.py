import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import onnxruntime as rt
import os
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
# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

# Load the YOLOv5 ONNX model.
model = rt.InferenceSession('D:/dh/nam3/hk1/xu_li_anh/WebDemo/PhatHienTraiCay_Yolo5_onnx/trai_cay.onnx')

def read_class_names(file_path):
    with open(file_path, 'r') as file:
        class_names = file.read().strip().split('\n')
    return class_names

model_files_dir = os.path.dirname('D:/dh/nam3/hk1/xu_li_anh/WebDemo/PhatHienTraiCay_Yolo5_onnx/trai_cay.onnx')
class_names = read_class_names(os.path.join(model_files_dir, "D:/dh/nam3/hk1/xu_li_anh/WebDemo/PhatHienTraiCay_Yolo5_onnx/trai_cay.names"))

def draw_label(im, label, x, y):
    """Draw text onto image at location."""

    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]),FONT_FACE, FONT_SCALE, YELLOW,THICKNESS, cv2.LINE_AA)

def post_process(input_image, outputs):
    class_ids, confidences, boxes = [], [], []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    result_image = input_image.copy()

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(result_image, (left, top), (left + width, top + height),BLUE, 3*THICKNESS)
        label = f"{class_names[class_ids[i]]}:{confidences[i]:.2f}"
        draw_label(result_image, label, left, top)

    return result_image

st.title("Phát Hiện Trái Cây - YOLOv5")

uploaded_image = st.file_uploader("Tải lên ảnh", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)

    st.image(image, caption="Ảnh Được Tải Lên", use_column_width=True)

    if st.button("Phân Loại"):
        # Pre-process the input image.
        blob = cv2.dnn.blobFromImage(image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

        # Run inference to get output of the output layers.
        inputs = {model.get_inputs()[0].name: blob}
        outputs = model.run(None, inputs)

        output_image = post_process(image, outputs)
        st.image(output_image, caption="Ảnh Đã Được Phân Loại", use_column_width=True)
