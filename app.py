import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
)
from tensorflow.keras.models import load_model
from src.face_rec import Face_recognition
face_recognition = Face_recognition()


from config import (
    IMAGE_EXAMPLE,
    IMAGE_TYPES,
    MODEL_CONFIG,
    MODEL_FACE_DETECTOR,
    MODEL_MASK_DETECTOR,
    RTC_CONFIGURATION,
)
# from sources.detect_mask_image import detect_mask_in_image

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(
    page_title="Employee Auto Checking",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load models
# prototxtPath = MODEL_CONFIG
# weightsPath = MODEL_FACE_DETECTOR
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# maskNet = load_model(MODEL_MASK_DETECTOR)
# example_image = IMAGE_EXAMPLE
# print("[INFO] loaded face mask detector model")..


class VideoProcessor:
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = face_recognition.recogny_face(image)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def local_css(file_name):
    # Method for reading styles.css and applying necessary changes to HTML
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# def choose_image():
#     # st.sidebar.markdown('Upload your image ⬇')
#     image_file = st.sidebar.file_uploader("", type=IMAGE_TYPES)

#     if not image_file:
#         text = """This is a detection example.
#         Try your input from the left sidebar.
#         """
#         st.markdown(
#             '<h6 align="center">' + text + "</h6>",
#             unsafe_allow_html=True,
#         )
#         st.image(example_image, use_column_width=True)
#     else:
#         st.sidebar.markdown(
#             "__Image is uploaded successfully!__",
#             unsafe_allow_html=True,
#         )
#         st.markdown(
#             '<h4 align="center">Detection result</h4>',
#             unsafe_allow_html=True,
#         )

#         PIL_image = Image.open(image_file)

#         image = np.array(PIL_image)
#         image = detect_mask_in_image(image, faceNet, maskNet)
#         st.image(image, use_column_width=True)


def choose_webcam():
    st.sidebar.markdown('Click "START" to connect this app to a server')
    st.sidebar.markdown("It may take a minute, please wait...")
    webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )


def main():
    local_css("styles.css")
    st.markdown(

        '<h6 align="center"><img src="https://www.tokyotechlab.com/wp-content/uploads/2019/12/ttlab-logo.svg" alt="TTL" width="160" height="150">TTLAB-AI-TEAM</h6>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 align="center">😷 Auto Checking Employee TTLAB</h1>',
        unsafe_allow_html=True,
    )
    st.set_option("deprecation.showfileUploaderEncoding", False)
    choice = st.sidebar.radio(
        "Select an input option:",
        ["Image", "Webcam"],
    )
    # if choice == "Image":
    #     choose_image()

    if choice == "Webcam":
        choose_webcam()


if __name__ == "__main__":
    main()