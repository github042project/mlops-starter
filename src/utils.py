# src/utils.py
import os
import joblib

def save_model(model, path="app/model_store/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path="app/model_store/model.joblib"):
    return joblib.load(path)
