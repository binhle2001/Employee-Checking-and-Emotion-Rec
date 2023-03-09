import cv2
import numpy as np
from openvino.inference_engine import IECore
from src.settings import (
    MODEL_IDENTIFY_XML,
    MODEL_IDENTIFY_BIN,
    FACE_SIZE,
)

class Face_identifier:
    def __init__(self) -> None:
        
# Load the Inference Engine API
        self.ie = IECore()

        # Load the ReID model
        self.model_xml = MODEL_IDENTIFY_XML
        self.model_bin = MODEL_IDENTIFY_BIN
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)

        # Load the model into the inference engine
        self.exec_net = self.ie.load_network(network=self.net, device_name="CPU")
        self.input_blob = next(iter(self.net.input_info))


    # Load an image and resize it
    def embed_image(self, image: np.ndarray):
        image = cv2.resize(image, (FACE_SIZE, FACE_SIZE))

        # Prepare the image for inference
        image = image.transpose((2, 0, 1))
        input_data = np.expand_dims(image, axis=0)

        # Run inference on the image
        output = self.exec_net.infer(inputs={self.input_blob: image})
        output_blob = next(iter(output))
        output_data = output[output_blob][0]
        output_data = output_data.flatten()
        return output_data

    def result_name(self, image, data_train, classes, threshold=0.7):
        image_embedding = self.embed_image(image)
        for i in range(len(classes)):
            data_train[i] = np.array(data_train[i])
            cosine = calculate_cosine_similarity(image_embedding, data_train[i])
            if abs(cosine) >= threshold:
                return classes[i]
        return "Unknown"
def calculate_cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))