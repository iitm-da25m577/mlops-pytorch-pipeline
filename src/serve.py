"""FastAPI inference server for the Fashion-MNIST classifier."""
import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
])

state = {"model": None, "device": None}


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = get_model(
        architecture=checkpoint.get("architecture", "cnn"),
        num_classes=checkpoint.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"], state["device"] = load_model()
    except FileNotFoundError:
        state["model"], state["device"] = None, None
    yield


app = FastAPI(title="Fashion-MNIST Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    tensor = _preprocess(img).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        logits = state["model"](tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

    return {
        "predictions": [
            {"class": CLASS_NAMES[i], "probability": round(p, 4)}
            for i, p in enumerate(probs)
        ],
        "predicted_class": CLASS_NAMES[int(torch.tensor(probs).argmax())],
    }
