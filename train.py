import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# ✅ Embedded dataset
data = {
    "size": [2104, 1600, 2400, 1416, 3000],
    "bedrooms": [3, 3, 3, 2, 4],
    "price": [399900, 329900, 369000, 232000, 539900]
}
df = pd.DataFrame(data)
X = df[["size", "bedrooms"]]
y = df["price"]

# ✅ Train model
model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)
mse = mean_squared_error(y, preds)

# ✅ MLflow logging
mlflow.set_experiment("HousePricePredictor")
with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

# ✅ Save model
joblib.dump(model, "model.pkl")
