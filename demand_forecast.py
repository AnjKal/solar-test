import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("solar_microgrid_dataset.csv")

# -----------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------
df["time_slot"] = df["time_slot"].map({
    "Night": 0,
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3
})

df["weather"] = df["weather"].map({
    "Clear": 0,
    "Cloudy": 1,
    "Rainy": 2
})

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df[
    ["hour", "time_slot", "temperature_c", "weather"]
]
y = df["adjusted_load_kw"]

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Demand Forecasting Results")
print("----------------------------")
print(f"MAE  : {mae:.2f} kW")
print(f"RMSE : {rmse:.2f} kW")

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label="Actual Demand", marker="o")
plt.plot(y_pred[:50], label="Predicted Demand", marker="x")
plt.xlabel("Sample Index")
plt.ylabel("Demand (kW)")
plt.title("Actual vs Predicted Demand")
plt.legend()
plt.grid()
plt.show()
