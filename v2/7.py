"""Stress evaluation dataset generator used by the microgrid scheduler.

`microgrid_scheduler.py` calls this module as a script (`python 7.py`).
To remain compatible, this script writes the dataset to
`next_month_7_day_forecast.csv`, which is then consumed by `ppo_inference.py`.

This version generates a *one-day* (24-hour) stress scenario with intermittent
high-blackout periods and maintenance windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_CSV = Path(__file__).with_name("next_month_7_day_forecast.csv")


def _hourly_grid_regime(hour: int) -> tuple[int, float, int, int, float]:
    """Return (grid_status, blackout_probability, blackout_type, maintenance_active, outage_min)."""

    # High-blackout windows (>=30% of hours), about half maintenance.
    # Maintenance must imply grid down.
    if hour in {3, 4, 13, 14}:  # 4/24 hours maintenance (type 2)
        return 0, 0.95, 2, 1, 240.0

    if hour in {6, 7, 10, 18}:  # 4/24 hours high blackout without maintenance
        # Mix: some risk-only (grid up) and some actual outage (grid down)
        if hour in {6, 18}:
            return 1, 0.70, 1, 0, 60.0
        return 0, 0.85, 1, 0, 180.0

    # Normal
    return 1, 0.05, 0, 0, 0.0


def generate_stress_evaluation_7_days(
    start_date: str = "2024-06-01",
    save_path: Union[str, Path] = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """Generate a one-day stress-evaluation dataset.

    Notes:
    - The function name is kept for backward compatibility.
    - Output columns match `MicrogridEnv.observation_columns` expectations.
    """

    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    battery_capacity_kwh = 20.0
    battery_max_charge_kw = 5.0
    battery_max_discharge_kw = 5.0

    for h in range(24):
        ts = start + timedelta(hours=h)

        hour = ts.hour
        day = 0  # single-day scenario

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        is_weekend = 1 if ts.weekday() >= 5 else 0

        # Solar (diurnal)
        daylight = np.sin(np.pi * np.clip((hour - 6) / 12, 0.0, 1.0))
        solar_peak = 6.5
        solar_actual = max(0.0, float(daylight * solar_peak))
        solar_f1 = max(0.0, float(solar_actual * 0.85))
        solar_f4 = max(0.0, float(solar_actual * 1.15))

        # Loads (simple daily pattern)
        evening_bump = 1.35 if 18 <= hour <= 22 else 1.0
        load_p1 = 1.6 * evening_bump
        load_p2 = 2.2 * evening_bump
        load_p3 = 2.6 * evening_bump

        # SOC starts high; the environment evolves SOC internally during inference.
        battery_soc = 0.85

        grid_status, blackout_probability, blackout_type, maintenance_active, outage_min = _hourly_grid_regime(hour)

        # Enforce implication rules
        if blackout_type == 2:
            maintenance_active = 1
            grid_status = 0
        if maintenance_active == 1:
            grid_status = 0

        expected_outage_duration_min = float(outage_min)
        is_islanded = int(grid_status == 0 and maintenance_active == 0)
        available_island_power_kw = float(solar_actual + battery_max_discharge_kw)

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
    print(f"Stress evaluation dataset saved: {save_path}")
    return df


if __name__ == "__main__":
    df = generate_stress_evaluation_7_days(save_path=DEFAULT_OUTPUT_CSV)
    print(f"Generated stress dataset with {len(df)} rows at {DEFAULT_OUTPUT_CSV}")
