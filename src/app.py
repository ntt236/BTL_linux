import os, json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from schemas import PredictReq, PredictRes
from model import load_model, predict_proba

APP_VERSION = os.getenv("APP_VERSION", "dev")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_latest.joblib")
MODEL_META = os.getenv("MODEL_META", "")  # optional: meta json path

app = FastAPI(title="VKU MLOps Predictor", version=APP_VERSION)

_model, _features = load_model(MODEL_PATH)

def label_from_percent(p: float) -> str:
    if p >= 75: return "CAO"
    if p >= 45: return "TRUNG_BÌNH"
    return "THẤP"

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok", "app_version": APP_VERSION, "model_path": MODEL_PATH}

@app.get("/model-info")
def model_info():
    info = {"app_version": APP_VERSION, "model_path": MODEL_PATH}
    if MODEL_META and os.path.exists(MODEL_META):
        try:
            info["meta"] = json.loads(open(MODEL_META, "r", encoding="utf-8").read())
        except Exception:
            info["meta"] = "unreadable"
    return info

@app.post("/predict", response_model=PredictRes)
def predict(req: PredictReq):
    feats = [req.math, req.physics, req.chemistry, req.english, req.priority]
    prob = predict_proba(_model, feats)
    percent = round(prob * 100, 2)
    return PredictRes(
        probability=prob,
        percent=percent,
        label=label_from_percent(percent),
        model_version=APP_VERSION
    )
