import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "historical_sales_data.csv")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("Dataset Preview:")
print(df.head())

df["MonthIndex"] = range(len(df))

X = df[["MonthIndex"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

print("\nModel Evaluation:")
print("MAE:", mae)
print("R2 Score:", r2)

future = pd.DataFrame({"MonthIndex": [24, 25, 26, 27, 28, 29]})
future_predictions = model.predict(future)

forecast_df = pd.DataFrame({
    "FutureMonth": ["2026-01","2026-02","2026-03","2026-04","2026-05","2026-06"],
    "PredictedSales": future_predictions
})

forecast_df.to_csv(os.path.join(BASE_DIR, "data", "sales_forecast.csv"), index=False)

joblib.dump(model, os.path.join(MODEL_DIR, "sales_forecast_model.pkl"))

plt.figure(figsize=(8,5))
plt.plot(df["MonthIndex"], y, label="Historical Sales")
plt.plot(df["MonthIndex"], predictions, label="Model Prediction")
plt.legend()
plt.title("Historical Sales vs Prediction")
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOT_DIR, "historical_vs_prediction.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.plot(forecast_df["FutureMonth"], forecast_df["PredictedSales"])
plt.title("Future Sales Forecast")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOT_DIR, "sales_forecast.png"))
plt.close()

print("\nForecast:")
print(forecast_df)

print("\nProject completed successfully.")