from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from model import load_artifact
from schemas import BatteryFeatures

app = FastAPI(title="EV Battery Failure Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = load_artifact()

BASE_DIR = os.path.dirname(__file__)


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: BatteryFeatures):
    result = predictor.predict(features.dict())
    return result
