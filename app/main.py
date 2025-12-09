# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils import load_model
import numpy as np
import os
from typing import List

app = FastAPI(title="MLops Starter API")

class Instance(BaseModel):
    features: List[float]

# try to load model at startup
MODEL_PATH = os.environ.get("MODEL_PATH", "app/model_store/model.joblib")
model = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("Warning: model not found at startup:", e)
    model = None

@app.get("/")
def root():
    return {"message": "Hello! Model Serving API is running."}

@app.post("/predict")
def predict(inst: Instance):
    global model
    if model is None:
        return {"error": "model not loaded"}
    x = np.array(inst.features).reshape(1, -1)
    pred = model.predict(x)[0]
    proba = model.predict_proba(x).max()
    return {"prediction": int(pred), "confidence": float(proba)}
