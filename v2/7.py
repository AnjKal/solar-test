"""
Randomized stress evaluation dataset generator for microgrid scheduler.

- No hard-coded hours
- Diurnal + stochastic solar
- Probabilistic grid outages and maintenance
- Designed to explore all possible system states
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_CSV = Path(__file__).with_name("next_month_7_day_forecast.csv")

rng = np.random.default_rng(42)  # reproducible randomness


def generate_random_stress_day(
    start_date: str = "2024-06-01",
    save_path: Union[str, Path] = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:

    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    # Battery constants
    battery_capacity_kwh = 20.0
    battery_max_charge_kw = 5.0
    battery_max_discharge_kw = 5.0

    # Solar envelope parameters
    sunrise_mean = 6
    sunset_mean = 18
    solar_peak_mean = 6.5
    solar_peak_std = 1.5

    # Load base means
    base_p1 = 1.5
    base_p2 = 2.2
    base_p3 = 2.7

    for h in range(24):
        ts = start + timedelta(hours=h)
        hour = ts.hour

        # -----------------------------
        # Time encodings
        # -----------------------------
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = 0.0
        day_cos = 1.0
        is_weekend = int(ts.weekday() >= 5)

        # -----------------------------
        # Solar generation (randomized bell)
        # -----------------------------
        sunrise = sunrise_mean + rng.normal(0, 0.5)
        sunset = sunset_mean + rng.normal(0, 0.5)

        if hour < sunrise or hour > sunset:
            solar_actual = 0.0
        else:
            solar_peak = max(2.0, rng.normal(solar_peak_mean, solar_peak_std))
            x = (hour - sunrise) / (sunset - sunrise)
            solar_actual = solar_peak * np.sin(np.pi * x)
            solar_actual = max(0.0, solar_actual + rng.normal(0, 0.4))

        solar_f1 = max(0.0, solar_actual * rng.uniform(0.7, 0.95))
        solar_f4 = max(0.0, solar_actual * rng.uniform(1.05, 1.3))

        # -----------------------------
        # Loads (priority-based stochastic)
        # -----------------------------
        evening_bias = rng.uniform(1.1, 1.4) if 17 <= hour <= 22 else 1.0

        load_p1 = max(0.3, rng.normal(base_p1 * evening_bias, 0.25))
        load_p2 = max(0.5, rng.normal(base_p2 * evening_bias, 0.35))
        load_p3 = max(0.8, rng.normal(base_p3 * evening_bias, 0.45))

        # -----------------------------
        # Battery SOC (initial snapshot only)
        # -----------------------------
        battery_soc = rng.uniform(0.25, 0.95)

        # -----------------------------
        # Grid + outage logic (random)
        # -----------------------------
        blackout_probability = rng.uniform(0.0, 0.9)

        maintenance_active = int(rng.random() < 0.15)   # 15% chance
        forced_blackout = rng.random() < blackout_probability

        if maintenance_active:
            grid_status = 0
            blackout_type = 2
            outage_min = rng.uniform(120, 360)
        elif forced_blackout:
            grid_status = 0
            blackout_type = 1
            outage_min = rng.uniform(30, 240)
        else:
            grid_status = 1
            blackout_type = 0
            outage_min = 0.0

        expected_outage_duration_min = float(outage_min)

        # -----------------------------
        # Islanding logic
        # -----------------------------
        is_islanded = int(grid_status == 0 and maintenance_active == 0)
        available_island_power_kw = solar_actual + battery_max_discharge_kw

        records.append([
            ts,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            is_weekend,
            solar_actual,
            solar_f1,
            solar_f4,
            load_p1,
            load_p2,
            load_p3,
            battery_soc,
            battery_capacity_kwh,
            battery_max_charge_kw,
            battery_max_discharge_kw,
            grid_status,
            blackout_probability,
            blackout_type,
            expected_outage_duration_min,
            maintenance_active,
            is_islanded,
            available_island_power_kw,
        ])

    df = pd.DataFrame(
        records,
        columns=[
            "timestamp",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "is_weekend",
            "solar_actual_kw",
            "solar_forecast_t+1_kw",
            "solar_forecast_t+4_kw",
            "load_p1_kw",
            "load_p2_kw",
            "load_p3_kw",
            "battery_soc",
            "battery_capacity_kwh",
            "battery_max_charge_kw",
            "battery_max_discharge_kw",
            "grid_status",
            "blackout_probability",
            "blackout_type",
            "expected_outage_duration_min",
            "maintenance_active",
            "is_islanded",
            "available_island_power_kw",
        ],
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Random stress dataset saved to {save_path}")

    return df


if __name__ == "__main__":
    df = generate_random_stress_day()
    print(df.describe())
