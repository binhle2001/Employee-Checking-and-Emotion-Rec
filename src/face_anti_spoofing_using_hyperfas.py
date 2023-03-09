import math
import cv2
import time
import numpy as np
import os
import keras
import tensorflow as tf
from src.settings import MODEL_FACE_ANTI_SPOOFING_HYPERFAS
model = keras.models.load_model(MODEL_FACE_ANTI_SPOOFING_HYPERFAS)
def detect_face_spoofing(face, threshold = 0.5):
    face = (cv2.resize(face,(224,224))-127.5)/127.5
    score = model.predict(np.array([face]))[0]
    score = float(score)
    if score > threshold:
        pred = 1
    else:
        pred = 0
    return pred, score