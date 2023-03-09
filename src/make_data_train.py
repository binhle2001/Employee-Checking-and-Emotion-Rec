import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from face_identification import Face_identifier
from face_detection import Face_detector
import os
import cv2
import pandas as pd
import numpy as np
from settings import (
    DATA_FACE_DIR,
)

class Trainer:
    def __init__(self):
        self.df = pd.read_json(DATA_FACE_DIR)
        self.data_image = self.df["face"].to_numpy().tolist()
        self.members = self.df["name"].to_numpy().tolist()
        self.face_identifier = Face_identifier()
        self.face_detector = Face_detector()
    def add_member(self, image, name):
        image = self.face_detector.detect_face(image)[0][0]
        image_embedding = self.face_identifier.embed_image(image)
        self.data_image.append(image_embedding)
        self.members.append(name)
        np.append(self.members, name)
        df = pd.DataFrame({"face": self.data_image, "name": self.members})
        df.to_json(DATA_FACE_DIR)


