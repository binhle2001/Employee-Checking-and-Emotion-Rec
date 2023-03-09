import os
from src.MobileNet import MobileNet
import time
import numpy as np
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import torch
import PIL.Image as Image
from src.settings import MODEL_FACE_ANTI_SPOOFING_MOBILENET

model_path = MODEL_FACE_ANTI_SPOOFING_MOBILENET
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = MobileNet(2).to(device)
net.load_state_dict(torch.load(model_path)['net_state_dict'])
net.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
])

def detect_face_spoofing(face):
    face = Image.fromarray(face.astype('uint8'))
    face = transform(face)
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        output = net(face)
        output = F.softmax(output).cpu().numpy()
    print(output)
    pred = np.argmax(output)
    score = output[0][pred]
    return pred, score    