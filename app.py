import streamlit as st
import numpy as np
from PIL import Image

from vision_tasks import (
    detect_face,
    detect_hands,
    recognize_gesture,
    detect_pose,
    remove_background
)


st.set_page_config(
    page_title="MediaPipe Vision Studio",
    page_icon="🎥",
    layout="wide"
)


st.title("🎥 MediaPipe Vision Studio")
st.write(
    "Simple Streamlit project that combines 5 easy MediaPipe Vision tasks."
)


task = st.sidebar.selectbox(
    "Choose Vision Task",
    [
        "Face Detection",
        "Hand Tracking",
        "Gesture Recognition",
        "Pose Detection",
        "Background Removal"
    ]
)


input_type = st.sidebar.radio(
    "Choose Input Type",
    ["Upload Image", "Camera Snapshot"]
)


image_file = None

if input_type == "Upload Image":
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

else:
    image_file = st.camera_input("Take a photo")


if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    image_rgb = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)

    if task == "Face Detection":
        output, message = detect_face(image_rgb)

    elif task == "Hand Tracking":
        output, message = detect_hands(image_rgb)

    elif task == "Gesture Recognition":
        output, message = recognize_gesture(image_rgb)

    elif task == "Pose Detection":
        output, message = detect_pose(image_rgb)

    elif task == "Background Removal":
        output, message = remove_background(image_rgb)

    with col2:
        st.subheader("Processed Image")
        st.image(output, use_container_width=True)

    st.success(message)

else:
    st.info("Upload an image or take a camera snapshot to start.")