import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
import time
from src.face_detection import Face_detector
from src.face_identification import Face_identifier
import cv2
import pandas as pd
import numpy as np
from src.settings import (
    DATA_FACE_DIR,
)
import src.face_anti_spoofing_using_hyperfas as face_anti_spoofing_using_hyperfas
import src.face_anti_spoofing_using_mobilenet as face_anti_spoofing_using_mobilenet
from .facial_emotion_recognition import (
    emotion_recogny_openvino,
    emotion_recogny_tensor,
)
# df = pd.read_json(DATA_FACE_DIR)
# data_faces = df["face"].to_numpy().tolist()
# members = df["name"].to_numpy().tolist()
class Face_recognition:
    def __init__(self) -> None:
        self.df = pd.read_json(DATA_FACE_DIR)
        self.data_faces = self.df["face"].to_numpy().tolist()
        self.members = self.df["name"].to_numpy().tolist()
        self.face_detector = Face_detector()
        self.face_identifier = Face_identifier()
    def recogny_face(self, image: np.ndarray):
        # self.frame = image
        imgs, x, y  = self.face_detector.detect_face(image)
        if len(imgs) == 0:
            return image
        for i in range(len(imgs)):
            xmin, xmax = x[i]
            ymin, ymax = y[i]
            self.bbox = [[xmin, ymin], [xmax, ymax]]
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            # pred, score = face_anti_spoofing_using_hyperfas.detect_face_spoofing(imgs[i])
            pred, score = face_anti_spoofing_using_mobilenet.detect_face_spoofing(imgs[i])
            # emotion = emotion_recogny_openvino(imgs[i])
            emotion = emotion_recogny_tensor(imgs[i])
            # if pred == 0:
            #     self.name_member = "Fake" + '_' + str(round(score, 2))
            # else:
                # self.name_member = "Real" + '_' + str(round(score, 2))
            self.name_member = self.face_identifier.result_name(imgs[i], self.data_faces, self.members)
            
            if self.name_member != "Unknown":
                self.current_time = datetime.now()
                image = cv2.putText(
                image,
                self.name_member + ' ' + emotion,
                (xmin, ymin),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
                break
        # if self.frame is None:
        #     self.frame = image
        return image
# def main(image):
#     imgs, x, y = face_detection.detect_face(image)
#     if len(imgs) == 0:
#         return None
#     for i in range(len(imgs)):
#         xmin, xmax = x[i]
#         ymin, ymax = y[i]
#         bbox = [[xmin, ymin], [xmax, ymax]]
#         frame = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
#         name_member = face_identification.result_name(imgs[i], data_faces, members)
#         if name_member != "Unknown":
#             current_time = datetime.now()
#             break
#     return frame, bbox, name_member, current_time
# if __name__ == "__main__":
#     main(image)
