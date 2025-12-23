import numpy as np
import pandas as pd
import random

# -----------------------------
# CONFIG
# -----------------------------
DAYS = 30
HOURS = 24
BATTERY_CAPACITY = 100  # kWh

data = []

# -----------------------------
# HELPERS
# -----------------------------
def time_slot(hour):
    if hour < 6:
        return "Night"
    elif hour < 12:
        return "Morning"
    elif hour < 18:
        return "Afternoon"
    else:
        return "Evening"

def weather_condition():
    r = random.random()
    if r < 0.6:
        return "Clear"
    elif r < 0.85:
        return "Cloudy"
    else:
        return "Rainy"

# -----------------------------
# DATA GENERATION
# -----------------------------
battery_level = random.uniform(40, 80)

for day in range(1, DAYS + 1):
    for hour in range(HOURS):

        slot = time_slot(hour)
        weather = weather_condition()

        # Temperature curve
        temperature = 22 + 10 * np.sin((hour / 24) * np.pi)

        # Cloud cover
        cloud_cover = {
            "Clear": random.uniform(0.0, 0.3),
            "Cloudy": random.uniform(0.4, 0.7),
            "Rainy": random.uniform(0.8, 1.0)
        }[weather]

        # Solar generation
        if 6 <= hour <= 18:
            solar_kw = max(0, (1 - cloud_cover) * random.uniform(18, 28))
        else:
            solar_kw = 0

        # Demand pattern
        if slot in ["Morning", "Afternoon"]:
            base_load = random.uniform(35, 50)
        elif slot == "Evening":
            base_load = random.uniform(25, 35)
        else:
            base_load = random.uniform(15, 25)

        # Weather-based demand adjustment
        if temperature > 30:
            base_load += 5
        if weather == "Rainy":
            base_load += 3

        # Grid price
        grid_price = random.uniform(5, 7) if slot == "Night" else random.uniform(7, 10)

        # Grid availability
        grid_available = 1 if random.random() > 0.05 else 0

        # Battery update
        net_energy = solar_kw - base_load
        battery_level = np.clip(
            battery_level + net_energy * 0.1, 0, BATTERY_CAPACITY
        )

        data.append([
            day, hour, slot, temperature, cloud_cover, weather,
            round(solar_kw, 2),
            round(base_load, 2),
            round(base_load, 2),
            round(battery_level, 2),
            round(grid_price, 2),
            grid_available
        ])

# -----------------------------
# DATAFRAME
# -----------------------------
columns = [
    "day", "hour", "time_slot", "temperature_c", "cloud_cover", "weather",
    "solar_generation_kw",
    "base_load_kw", "adjusted_load_kw",
    "battery_level_kwh", "grid_price_rs", "grid_available"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("solar_microgrid_dataset.csv", index=False)

print("Solar-only dataset generated: solar_microgrid_dataset.csv")
print(df.head())
