import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="Nhận dạng chữ số viết tay", page_icon="🌐")
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

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callback for early stopping
early_stopping = keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

# Function to train the model
def train_model():
    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)

    # Save the model
    model.save('mnist_model.h5')

# Function to predict digits
def predict_digit(image):
    # Load pre-trained model
    model = load_model('mnist_model.h5')

    # Load and process the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        # Get bounding box
        (x, y, w, h) = cv2.boundingRect(contour)

        # Add padding to the bounding box
        padding = 10
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2

        # Extract ROI
        roi = threshold[y:y + h, x:x + w]

        # Resize ROI to 28x28 pixels
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize ROI
        roi = roi.astype('float32') / 255.0

        # Reshape ROI to match model input shape (1, 28, 28, 1)
        # Reshape ROI to match model input shape (1, 28, 28, 1)
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Make prediction
        prediction = model.predict(roi)
        digit = np.argmax(prediction)

        # Draw bounding box and predicted digit label
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Chuyển đổi ảnh OpenCV thành định dạng RGB để hiển thị trong Streamlit
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh và kết quả nhận dạng đối tượng sử dụng Streamlit
    st.image(img_rgb, channels="RGB")

# Create a sidebar with a button to train the model
if st.sidebar.button('Train Model'):
    train_model()

# Create a file uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Read the image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Perform prediction on the uploaded image
    predict_digit(image)