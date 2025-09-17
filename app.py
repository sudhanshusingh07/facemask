import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


# loading the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector.h5") 

model = load_model()
labels = ["with_mask", "without_mask"]


# using opencv’s face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# title and page name
st.set_page_config(page_title="Mask Detection App")
st.title("Mask Detection App")
st.write("Upload an image, and the app will detect faces with/without masks.")


# upload image section
uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # converting uploaded image into numpy array
    image = Image.open(uploaded_image)
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        # resizing face to model input size
        face_resized = cv2.resize(face, (128, 128))  
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        # predicting mask or no mask
        pred = model.predict(face_array, verbose=0)[0]
        label = labels[np.argmax(pred)]
        color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)

        # draw rectangle and label on image
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # show the final output image
    st.markdown("### Result")
    st.image(img, use_container_width=True)
else:
    st.info("Please upload an image to test mask detection.")


# how to use
with st.expander("How to Use"):
    st.markdown("""
    1. Upload an **image** (`.png`, `.jpg`, `.jpeg`)  
    2. App will detect faces and classify as:  
       - `with_mask`  
       - `without_mask`  
    """)
