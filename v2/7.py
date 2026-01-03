"""Utility for generating 7-day microgrid forecasts used by the Streamlit app.

This module wraps the logic that previously lived in 7.ipynb so it can be reused
programmatically by the application and the notebook.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DEFAULT_SOURCE_CSV = Path(__file__).with_name("solar_microgrid_dataset.csv")
DEFAULT_OUTPUT_CSV = Path(__file__).with_name("next_month_7_day_forecast.csv")


def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with engineered features needed for linear models."""

    working = df.copy()

    working["weather_code"] = working["weather"].map({
        "Clear": 0,
        "Cloudy": 1,
        "Rainy": 2,
    })

    working["hour_sin"] = np.sin(2 * np.pi * working["hour"] / 24)
    working["hour_cos"] = np.cos(2 * np.pi * working["hour"] / 24)

    working["solar_t+1"] = working["solar_generation_kw"].shift(-1)
    working["solar_t+4"] = working["solar_generation_kw"].shift(-4)
    working["load_t+1"] = working["adjusted_load_kw"].shift(-1)
    working["load_t+4"] = working["adjusted_load_kw"].shift(-4)

    working = working.dropna().reset_index(drop=True)
    return working


def generate_microgrid_forecast(
    source_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    *,
    battery_capacity_kwh: float = 100.0,
    battery_max_charge_kw: float = 5.0,
    battery_max_discharge_kw: float = 5.0,
    write_csv: bool = True,
) -> pd.DataFrame:
    """Create the 7-day forecast dataset and optionally persist it to disk."""

    source_csv = Path(source_csv or DEFAULT_SOURCE_CSV)
    output_csv = Path(output_csv or DEFAULT_OUTPUT_CSV)

    raw_df = pd.read_csv(source_csv)
    prepared_df = _prepare_training_frame(raw_df)

    feature_cols = [
        "hour",
        "hour_sin",
        "hour_cos",
        "temperature_c",
        "cloud_cover",
        "weather_code",
        "grid_available",
        "battery_level_kwh",
    ]

    X = prepared_df[feature_cols]

    solar_t1_model = LinearRegression().fit(X, prepared_df["solar_t+1"])
    solar_t4_model = LinearRegression().fit(X, prepared_df["solar_t+4"])
    load_t1_model = LinearRegression().fit(X, prepared_df["load_t+1"])
    load_t4_model = LinearRegression().fit(X, prepared_df["load_t+4"])

    hourly_means = prepared_df.groupby("hour").mean(numeric_only=True)

    future_rows = []
    for day in range(1, 8):
        for hour in range(24):
            ref = hourly_means.loc[hour]
            future_rows.append({
                "day": day,
                "hour": hour,
                "temperature_c": ref["temperature_c"],
                "cloud_cover": ref["cloud_cover"],
                "weather_code": round(ref["weather_code"]),
                "grid_available": 1,
                "battery_level_kwh": ref["battery_level_kwh"],
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "day_sin": np.sin(2 * np.pi * day / 365),
                "day_cos": np.cos(2 * np.pi * day / 365),
                "is_weekend": int(((day - 1) % 7) in [5, 6]),
                "solar_actual_kw": ref["solar_generation_kw"],
                "load_actual_kw": ref["adjusted_load_kw"],
            })

    future_df = pd.DataFrame(future_rows)

    results = []
    for _, row in future_df.iterrows():
        X_row = row[feature_cols].values.reshape(1, -1)

        solar_forecast_t1 = max(0.0, float(solar_t1_model.predict(X_row)[0]))
        solar_forecast_t4 = max(0.0, float(solar_t4_model.predict(X_row)[0]))

        load_forecast_t1 = max(0.0, float(load_t1_model.predict(X_row)[0]))
        load_forecast_t4 = max(0.0, float(load_t4_model.predict(X_row)[0]))

        load_total = row["load_actual_kw"]
        load_p1 = 0.40 * load_total
        load_p2 = 0.30 * load_total
        load_p3 = 0.30 * load_total

        load_p1_f1 = 0.40 * load_forecast_t1
        load_p2_f1 = 0.30 * load_forecast_t1
        load_p3_f1 = 0.30 * load_forecast_t1

        load_p1_f4 = 0.40 * load_forecast_t4
        load_p2_f4 = 0.30 * load_forecast_t4
        load_p3_f4 = 0.30 * load_forecast_t4

        blackout = 0.0
        if row["grid_available"] == 0:
            blackout += 0.4
        if solar_forecast_t1 < load_forecast_t1:
            blackout += 0.4
        if row["battery_level_kwh"] < 20:
            blackout += 0.2
        blackout = min(1.0, blackout)

        outage = 0 if blackout < 0.3 else 10 if blackout < 0.6 else 30
        blackout_type = 1 if blackout >= 0.85 else 0
        maintenance_active = 0
        grid_status = int(row["grid_available"])
        is_islanded = int(grid_status == 0 and maintenance_active == 0)

        battery_soc = float(np.clip(row["battery_level_kwh"] / battery_capacity_kwh, 0.0, 1.0))
        available_island_power = row["solar_actual_kw"] + battery_max_discharge_kw

        results.append({
            "hour_sin": row["hour_sin"],
            "hour_cos": row["hour_cos"],
            "day_sin": row["day_sin"],
            "day_cos": row["day_cos"],
            "is_weekend": row["is_weekend"],
            "solar_actual_kw": row["solar_actual_kw"],
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
            "battery_capacity_kwh": battery_capacity_kwh,
            "battery_max_charge_kw": battery_max_charge_kw,
            "battery_max_discharge_kw": battery_max_discharge_kw,
            "grid_status": grid_status,
            "blackout_probability": blackout,
            "blackout_type": blackout_type,
            "expected_outage_duration_min": outage,
            "maintenance_active": maintenance_active,
            "is_islanded": is_islanded,
            "available_island_power_kw": available_island_power,
        })

    forecast_df = pd.DataFrame(results)

    if write_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(output_csv, index=False)

    return forecast_df


if __name__ == "__main__":
    df = generate_microgrid_forecast()
    print(f"Generated forecast with {len(df)} rows at {DEFAULT_OUTPUT_CSV}")
