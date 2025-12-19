from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import train_model, predict_prob

app = FastAPI(title="VKU Admission Predictor")

model = train_model("data.csv")

class PredictReq(BaseModel):
    math: float
    physics: float
    chemistry: float
    english: float
    priority: float = 0

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict(req: PredictReq):
    student = [req.math, req.physics, req.chemistry, req.english, req.priority]
    prob = predict_prob(model, student)
    return {"probability": prob, "percent": round(prob * 100, 2)}
