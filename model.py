import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ['math', 'physics', 'chemistry', 'english', 'priority']

def train_model(csv_path: str = "data.csv"):
    data = pd.read_csv(csv_path)
    X = data[FEATURES]
    y = data["result"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def predict_prob(model, student: list[float]) -> float:
    prob = model.predict_proba([student])[0][1]
    return float(prob)
