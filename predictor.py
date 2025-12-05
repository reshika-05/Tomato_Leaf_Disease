# predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import os
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load classes + remedies
JSON_PATH = os.path.join(os.path.dirname(__file__), "classes.json")
with open(JSON_PATH, "r") as f:
    data = json.load(f)
    CLASS_NAMES = data["classes"]
    REMEDIES = data["remedies"]

NUM_CLASSES = len(CLASS_NAMES)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tomato_disease_model.pth")

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        values, indices = torch.topk(probs, 3)

    return [
        {
            "class": CLASS_NAMES[idx],
            "confidence": float(score),
            "remedy": REMEDIES.get(CLASS_NAMES[idx])
        }
        for score, idx in zip(values, indices)
    ]