import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("data.csv")

X = data[['math','physics','chemistry','english','priority']]
y = data['result']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Dữ liệu thí sinh mới
student = [[8.0, 7.5, 7.0, 6.5, 1]]

prob = model.predict_proba(student)[0][1]
print(f"Khả năng trúng tuyển vào VKU: {prob*100:.2f}%")
