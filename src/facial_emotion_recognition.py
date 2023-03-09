from .settings import (
    MODEL_FACE_EMOTION_BIN,
    MODEL_FACE_EMOTION_XML
)
from openvino.inference_engine import IECore
import numpy as np
import cv2

import time
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
from src.face_detection import Face_detector
from src.settings import (
    MODEL_FACE_EMOTION_WEIGHT, 
    MODEL_FACE_EMOTION_STRUCTURE,
    )

model_tensor = model_from_json(open(MODEL_FACE_EMOTION_STRUCTURE, "r").read())
model_tensor.load_weights(MODEL_FACE_EMOTION_WEIGHT)

model_xml = MODEL_FACE_EMOTION_XML
model_bin = MODEL_FACE_EMOTION_BIN
ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="CPU")
input_blob = next(iter(net.input_info))
emotions_dict = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Angry"}

def emotion_recogny_openvino(img: np.ndarray):
    blob = cv2.dnn.blobFromImage(img, size=(64, 64), ddepth=cv2.CV_8U)
    output = exec_net.infer(inputs={input_blob: blob})
    predictions = output["prob_emotion"].flatten()
    predicted_emotion = emotions_dict[np.argmax(predictions)]
    return predicted_emotion

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def emotion_recogny_tensor(img: np.ndarray):
    detected_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_face = cv2.resize(detected_face, (48, 48))

    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255
    predictions = model_tensor.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    emotion = emotions[max_index]
    return emotion
