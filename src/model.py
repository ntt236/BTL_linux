from __future__ import annotations
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["math", "physics", "chemistry", "english", "priority"]

def train_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y = df["result"]
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model

def save_model(model, path: str):
    joblib.dump({"model": model, "features": FEATURES}, path)

def load_model(path: str):
    payload = joblib.load(path)
    return payload["model"], payload["features"]

def predict_proba(model, features: list[float]) -> float:
    return float(model.predict_proba([features])[0][1])
