import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO

# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------
# Load Model
# -----------------------------------

model = models.resnet18(pretrained=False)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)

# -----------------------------------
# Download Model Weights
# -----------------------------------

MODEL_URL = "https://huggingface.co/the-kshitij-chauhan/Parking-Slot-Model/resolve/main/parking_resnet18.pth"

response = requests.get(MODEL_URL)

model_weights = BytesIO(response.content)

model.load_state_dict(
    torch.load(
        model_weights,
        map_location=device
    )
)

model = model.to(device)

model.eval()

# -----------------------------------
# Image Transform
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------
# Class Labels
# -----------------------------------

classes = {
    0: "Occupied",
    1: "Empty"
}

# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_slot(slot_image):

    # Convert numpy image to PIL
    pil_image = Image.fromarray(slot_image)

    image_tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        _, predicted = torch.max(outputs, 1)

        prediction = classes[
            predicted.item()
        ]

    return prediction