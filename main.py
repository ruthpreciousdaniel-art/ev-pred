import numpy as np
import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.model import load_artifact

app = FastAPI(title="EV Battery Failure Predictor API")
templates = Jinja2Templates(directory="app/templates")

model, scaler, label_encoders = load_artifact()


def predict(data: dict) -> dict:
  x = np.array([[features[f] for in FEATURE_ORDER]])
  x = scaler.transform(x)
  x = torch.tensor(x_scaled, dtype=torch.float32)

  with torch.no_grad():
    logit = model(x).squeeze(1)
    prob = torch.sigmoid(logit).item()

  pred_class = label_encoder.inverse_transform([int(prob >= 0.5)])[0]
  return pred_class, prob


@app.get("/")
def serve_ui():
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: BatteryFeatures):
    result = predictor.predict(features.dict())
    return result
