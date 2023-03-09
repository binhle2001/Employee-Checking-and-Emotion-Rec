import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime
import time
from face_detection import Face_detector
from face_identification import Face_identifier
import cv2
import pandas as pd
from settings import (
    DATA_FACE_DIR,
    DATA_RESULT_DIR,
)

# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


cap = cv2.VideoCapture(0)  # 0 is source of webcam


def main():
    # TODO Tracking face to optimize
    face_detector = Face_detector()
    face_identifier = Face_identifier()
    name_current = []
    df = pd.read_json(DATA_FACE_DIR)

    data_image = df["face"].to_numpy().tolist()
    members = df["name"].to_numpy().tolist()
    time_current = time.time()
    time_put_text = []
    while True:
        df2 = pd.read_json(DATA_RESULT_DIR)
        name = df2["name"].to_numpy().tolist()
        time_full = df2["time"].to_numpy().tolist()
        ret, frame = cap.read()
        faces, x, y = face_detector.detect_face(frame)
        if len(faces) == 0:
            continue
        for i in range(len(faces)):
            xmin, xmax = x[i]
            ymin, ymax = y[i]
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            data_faces = data_image
            name_member = face_identifier.result_name(faces[i], data_faces, members)
            current_time = datetime.now()
            if time.time() - time_current >= 3600:
                df = pd.read_json(DATA_FACE_DIR)
                data_image = df["face"].to_numpy().tolist()
                members = df["name"].to_numpy().tolist()
            if name_member != "Unknown":
                name_current.append(name_member)
                time_put_text.append(time.time())
                df = df[df["name"] != name_member]
                data_image = df["face"].to_numpy().tolist()
                members = df["name"].to_numpy().tolist()
                name.append(name_member)
                current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                time_full.append(current_time)
                info = pd.DataFrame(
                    {
                        "name": name,
                        "time": time_full,
                    }
                )
                info.to_json(DATA_RESULT_DIR)
                break

        for name in name_current:
            frame = cv2.putText(
                frame,
                name,
                (50 * i, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            i += 1
        if len(time_put_text) != 0:
            if time.time() - time_put_text[0] >= 5:
                time_put_text.pop()
                name_current.pop()

        cv2.imshow("frame", frame)
        # break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
