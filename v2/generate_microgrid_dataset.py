import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_microgrid_dataset(
    start_date: str = "2024-01-01",
    days: int = 365,
    seed: int = 42,
    save_path: str = "synthetic_microgrid_dataset.csv",
    *,
    p_high_blackout: float = 0.35,
    p_maintenance_given_high: float = 0.50,
    battery_capacity_kwh: float = 20.0,
    battery_max_charge_kw: float = 5.0,
    battery_max_discharge_kw: float = 5.0,
) -> pd.DataFrame:
    """Generate a synthetic training dataset aligned with MicrogridEnv + 7.py.

    Hard requirements enforced by construction:
    - >=30% rows with blackout_probability > 0.6 (default p_high_blackout=0.35)
    - About half of high-blackout rows are maintenance (blackout_type==2)
    - blackout_type==2 => maintenance_active==1 and grid_status==0
    - maintenance_active==1 => grid_status==0
    - Some rows have high blackout + maintenance co-occurring (maintenance rows are high-blackout)
    """

    np.random.seed(seed)
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    for d in range(days):
        for h in range(24):
            ts = start + timedelta(days=d, hours=h)

            # ---------------------------
            # TIME FEATURES
            # ---------------------------
            hour_sin = np.sin(2 * np.pi * h / 24)
            hour_cos = np.cos(2 * np.pi * h / 24)
            day_sin = np.sin(2 * np.pi * d / 365)
            day_cos = np.cos(2 * np.pi * d / 365)
            is_weekend = 1 if ts.weekday() >= 5 else 0

            # ---------------------------
            # SOLAR (diurnal)
            # ---------------------------
            # Daylight approx: 06:00-18:00
            daylight = np.sin(np.pi * np.clip((h - 6) / 12, 0.0, 1.0))
            solar_peak = np.random.uniform(4.0, 8.0)
            solar_actual = max(0.0, float(daylight * solar_peak * np.random.uniform(0.8, 1.2)))
            solar_f1 = max(0.0, float(solar_actual * np.random.uniform(0.7, 1.3)))
            solar_f4 = max(0.0, float(solar_actual * np.random.uniform(0.6, 1.4)))

            # ---------------------------
            # LOAD (daily pattern)
            # ---------------------------
            evening_bump = 1.0 + (0.35 if 18 <= h <= 22 else 0.0)
            base = np.random.uniform(0.8, 1.6) * evening_bump
            load_p1 = float(np.random.uniform(0.8, 2.2) * base)
            load_p2 = float(np.random.uniform(0.6, 2.8) * base)
            load_p3 = float(np.random.uniform(0.4, 3.2) * base)

            # Occasional overload
            if np.random.rand() < 0.10:
                scale = float(np.random.uniform(1.2, 1.7))
                load_p1 *= scale
                load_p2 *= scale
                load_p3 *= scale

            # ---------------------------
            # BATTERY
            # ---------------------------
            battery_soc = float(np.random.uniform(0.10, 0.90))

            # ---------------------------
            # BLACKOUT / MAINTENANCE
            # ---------------------------
            high_blackout = np.random.rand() < p_high_blackout

            if high_blackout:
                is_maintenance = np.random.rand() < p_maintenance_given_high
                blackout_probability = float(np.random.uniform(0.65, 1.0))

                if is_maintenance:
                    blackout_type = 2
                    maintenance_active = 1
                    grid_status = 0
                    expected_outage_duration_min = float(np.random.uniform(180, 480))
                else:
                    blackout_type = 1
                    maintenance_active = 0

                    # Mix: some true outage (grid down), some preemptive risk (grid up)
                    grid_status = int(np.random.rand() < 0.50)
                    if grid_status == 0:
                        expected_outage_duration_min = float(np.random.uniform(30, 240))
                    else:
                        expected_outage_duration_min = float(np.random.uniform(10, 180))
            else:
                maintenance_active = 0

                # Mostly normal operation, occasional medium-risk events
                if np.random.rand() < 0.15:
                    blackout_type = 1
                    blackout_probability = float(np.random.uniform(0.35, 0.60))
                    grid_status = int(np.random.rand() < 0.95)
                    expected_outage_duration_min = float(np.random.uniform(0, 60))
                else:
                    blackout_type = 0
                    blackout_probability = float(np.random.uniform(0.0, 0.30))
                    grid_status = 1
                    expected_outage_duration_min = 0.0

            # Enforce implication rules
            if blackout_type == 2:
                maintenance_active = 1
                grid_status = 0
            if maintenance_active == 1:
                grid_status = 0

            is_islanded = int(grid_status == 0 and maintenance_active == 0)
            available_island_power_kw = float(solar_actual + battery_max_discharge_kw)

            records.append(
                [
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
                ]
            )

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

    df.to_csv(save_path, index=False)
    print(f"✅ Synthetic microgrid dataset saved: {save_path}")
    return df


# Backward-compatible name (older docs referenced this symbol)
def generate_stress_training_dataset(
    start_date="2024-01-01",
    days=365,
    seed=42,
    save_path="synthetic_microgrid_dataset.csv",
):
    return generate_synthetic_microgrid_dataset(
        start_date=start_date,
        days=days,
        seed=seed,
        save_path=save_path,
    )


if __name__ == "__main__":
    generate_synthetic_microgrid_dataset()
