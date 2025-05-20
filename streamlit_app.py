import streamlit as st
from PIL import Image
from corruption import corrupt_image
import io

st.title("Clear Vision")

uploaded_file = st.file_uploader("Upload a clean image", type=["jpg", "jpeg", "png"])
mode = st.selectbox("Choose corruption mode", ["noise", "banding", "mask", "All"])
submit = st.button("Enter")

# Load the image if a file is uploaded
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

# Show the original image before submission
if image is not None and not submit:
    st.image(image, caption="Original Image", use_container_width=True)

# On submit, show before and after side by side
if image is not None and submit:
    corrupted = corrupt_image(image, mode=mode)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Before", use_container_width=True)
    with col2:
        st.image(corrupted, caption="After", use_container_width=True)
