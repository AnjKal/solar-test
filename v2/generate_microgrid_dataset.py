import numpy as np
import pandas as pd
import os

# -----------------------------
# Configuration
# -----------------------------
NUM_DAYS = 60                 # number of simulated days
HOURS_PER_DAY = 24
SEED = 42

BATTERY_CAPACITY_KWH = 20.0
BATTERY_MAX_CHARGE_KW = 5.0
BATTERY_MAX_DISCHARGE_KW = 5.0

np.random.seed(SEED)

# -----------------------------
# Helper functions
# -----------------------------
def diurnal_solar_curve(hour, peak=10.4):
    """Smooth solar curve with peak at midday"""
    return max(
        0.0,
        peak * np.sin(np.pi * (hour - 6) / 12)
    )

def residential_load_curve(hour):
    """Morning + evening peaks"""
    if 6 <= hour <= 9:
        return 1.15
    elif 18 <= hour <= 22:
        return 1.25
    elif 0 <= hour <= 5:
        return 0.85
    else:
        return 1.0

# -----------------------------
# Dataset generation
# -----------------------------
rows = []

battery_soc = 0.6  # initial SOC

for day in range(NUM_DAYS):
    # day-level variability
    solar_scale = np.random.normal(1.0, 0.15)
    load_scale = np.random.normal(1.0, 0.08)

    # random maintenance window (rare)
    maintenance_day = np.random.rand() < 0.08
    maintenance_hours = np.random.choice(range(10, 16), size=4, replace=False) if maintenance_day else []

    for hour in range(HOURS_PER_DAY):

        # -----------------------------
        # Time encoding
        # -----------------------------
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 365)
        day_cos = np.cos(2 * np.pi * day / 365)
        is_weekend = int(day % 7 in [5, 6])

        # -----------------------------
        # Solar generation
        # -----------------------------
        solar_actual = diurnal_solar_curve(hour) * solar_scale
        solar_actual += np.random.normal(0, 0.6)   # cloud noise
        solar_actual = np.clip(solar_actual, 0, 10.5)

        solar_forecast_t1 = solar_actual + np.random.normal(0, 0.9)
        solar_forecast_t4 = solar_actual + np.random.normal(0, 1.4)

        solar_forecast_t1 = max(0, solar_forecast_t1)
        solar_forecast_t4 = max(0, solar_forecast_t4)

        # -----------------------------
        # Load demand
        # -----------------------------
        base_load = 6.05 * residential_load_curve(hour) * load_scale
        load_total = np.random.normal(base_load, 0.95)
        load_total = np.clip(load_total, 4.5, 7.9)

        # Priority split
        load_p1 = 0.40 * load_total
        load_p2 = 0.30 * load_total
        load_p3 = 0.30 * load_total

        # Forecasts
        load_p1_f1 = load_p1 + np.random.normal(0, 0.15)
        load_p2_f1 = load_p2 + np.random.normal(0, 0.12)
        load_p3_f1 = load_p3 + np.random.normal(0, 0.20)

        load_p1_f4 = load_p1 + np.random.normal(0, 0.25)
        load_p2_f4 = load_p2 + np.random.normal(0, 0.20)
        load_p3_f4 = load_p3 + np.random.normal(0, 0.30)

        # -----------------------------
        # Grid & outage modeling
        # -----------------------------
        maintenance_active = int(hour in maintenance_hours)
        blackout_probability = np.random.beta(2, 12)

        if maintenance_active:
            blackout_type = 2  # planned maintenance
            grid_status = 0
        elif blackout_probability > 0.85:
            blackout_type = 1  # unplanned
            grid_status = 0
        else:
            blackout_type = 0
            grid_status = 1

        expected_outage_duration = (
            np.random.randint(60, 180) if blackout_type == 1 else
            np.random.randint(120, 300) if blackout_type == 2 else
            0
        )

        # -----------------------------
        # Battery
        # -----------------------------
        battery_soc = np.clip(
            battery_soc + np.random.normal(0, 0.02),
            0.2,
            0.9
        )

        available_island_power = solar_actual + BATTERY_MAX_DISCHARGE_KW

        # -----------------------------
        # Islanding state (logged, not controlled here)
        # -----------------------------
        is_islanded = int(grid_status == 0 and not maintenance_active)

        # -----------------------------
        # Row assembly
        # -----------------------------
        rows.append({
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "is_weekend": is_weekend,

            "solar_actual_kw": solar_actual,
            "solar_forecast_t+1_kw": solar_forecast_t1,
            "solar_forecast_t+4_kw": solar_forecast_t4,

            "load_p1_kw": load_p1,
            "load_p2_kw": load_p2,
            "load_p3_kw": load_p3,

            "load_p1_forecast_t+1_kw": load_p1_f1,
            "load_p2_forecast_t+1_kw": load_p2_f1,
            "load_p3_forecast_t+1_kw": load_p3_f1,

            "load_p1_forecast_t+4_kw": load_p1_f4,
            "load_p2_forecast_t+4_kw": load_p2_f4,
            "load_p3_forecast_t+4_kw": load_p3_f4,

            "battery_soc": battery_soc,
            "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
            "battery_max_charge_kw": BATTERY_MAX_CHARGE_KW,
            "battery_max_discharge_kw": BATTERY_MAX_DISCHARGE_KW,

            "grid_status": grid_status,
            "blackout_probability": blackout_probability,
            "blackout_type": blackout_type,
            "expected_outage_duration_min": expected_outage_duration,

            "maintenance_active": maintenance_active,
            "is_islanded": is_islanded,
            "available_island_power_kw": available_island_power
        })

# -----------------------------
# Save dataset
# -----------------------------
df = pd.DataFrame(rows)

output_path = os.path.join(os.getcwd(), "synthetic_microgrid_dataset.csv")
df.to_csv(output_path, index=False)

print(f"Dataset generated: {output_path}")
print(df.describe())
