# Dataset Documentation

Project: Reinforcement Learning Microgrid Optimization (v2)

This document describes the datasets used by training, inference, and visualization components, their schemas, units, and the synthetic generation pipeline.

## Files
- Source synthetic dataset: [v2/synthetic_microgrid_dataset.csv](../v2/synthetic_microgrid_dataset.csv)
- 7-day forecast dataset: [v2/next_month_7_day_forecast.csv](../v2/next_month_7_day_forecast.csv)
- Generator script: [v2/generate_microgrid_dataset.py](../v2/generate_microgrid_dataset.py)

## Schema (shared core columns)
Columns common to both datasets (units in parentheses):
- `hour_sin` (–): sin(time-of-day); cyclic encoding
- `hour_cos` (–): cos(time-of-day); cyclic encoding
- `day_sin` (–): sin(day-of-year); cyclic encoding
- `day_cos` (–): cos(day-of-year); cyclic encoding
- `is_weekend` (0/1): weekend flag
- `solar_actual_kw` (kW): measured solar generation at timestep
- `solar_forecast_t+1_kw` (kW): solar forecast horizon +1
- `solar_forecast_t+4_kw` (kW): solar forecast horizon +4
- `load_p1_kw` (kW): priority-1 (critical) load demand
- `load_p2_kw` (kW): priority-2 load demand
- `load_p3_kw` (kW): priority-3 load demand
- `battery_soc` (0–1): battery state of charge fraction
- `battery_capacity_kwh` (kWh): usable battery capacity
- `battery_max_charge_kw` (kW): max feasible charge power
- `battery_max_discharge_kw` (kW): max feasible discharge power
- `grid_status` (0/1): 1=grid up, 0=grid down
- `blackout_probability` (0–1): probability of outage
- `blackout_type` (enum {0,1,2}): 0=none, 1=unplanned, 2=planned
- `expected_outage_duration_min` (minutes): expected outage duration if down
- `maintenance_active` (0/1): planned maintenance window flag
- `is_islanded` (0/1): islanded state indicator (logged)
- `available_island_power_kw` (kW): `solar_actual_kw + battery_max_discharge_kw`

## Extended forecast columns
- `load_p{1,2,3}_forecast_t+1_kw` (kW): per-load forecast horizon +1
- `load_p{1,2,3}_forecast_t+4_kw` (kW): per-load forecast horizon +4

## Synthetic Generation Pipeline
Implemented in [v2/generate_microgrid_dataset.py](../v2/generate_microgrid_dataset.py). Key components:

- Configuration:
  - `NUM_DAYS=365`, `HOURS_PER_DAY=24`, `SEED=42`
  - Battery params: `BATTERY_CAPACITY_KWH=20.0`, `BATTERY_MAX_CHARGE_KW=5.0`, `BATTERY_MAX_DISCHARGE_KW=5.0`

- Time encoding:
  - Cyclic features for hour and day; `is_weekend` from day-of-week.

- Solar generation:
  - Diurnal curve `diurnal_solar_curve(hour, peak=10.4) ≈ max(0, peak·sin(π·(hour−6)/12))`
  - Additive noise (cloud variability), scaling per day; clipped to [0, 10.5].
  - Forecasts `t+1`, `t+4` = `solar_actual` + Gaussian noise, clipped at 0.

- Load demand:
  - Base profile via `residential_load_curve` with morning/evening peaks.
  - Day scaling and Gaussian noise; clip total load to [4.5, 7.9].
  - Priority split: `p1=40%`, `p2=30%`, `p3=30%` of total.
  - Per-priority forecasts at horizons `t+1`, `t+4` with distinct noise levels.

- Grid/outage modeling:
  - Planned maintenance: ~20% of days get a 3-hour midday window; `maintenance_active=1` during that window.
  - Blackout probability mixture: nominal `Beta(2,12)`, with ~2% spikes (Uniform[0.85,1.0]).
  - Risk correlation: if `solar_actual < 0.6` and `load_total > 7.2`, add 0.15 to blackout probability (clipped).
  - `blackout_type`: 2 if maintenance; 1 if `blackout_probability > 0.85`; else 0.
  - `grid_status`: 0 when maintenance or blackout; else 1.
  - `expected_outage_duration_min`: sampled by outage type (0 if none).

- Battery modeling:
  - SOC is exogenous: random walk with noise, clipped to [0.2, 0.9].
  - `available_island_power_kw = solar_actual_kw + BATTERY_MAX_DISCHARGE_KW`.

- Islanding indicator:
  - `is_islanded = int(grid_status == 0 and not maintenance_active)` logged for reference.

- Output:
  - Saves to [v2/synthetic_microgrid_dataset.csv](../v2/synthetic_microgrid_dataset.csv) and prints `describe()`.

## Dataset Usage in Code
- Environment observations use the subset `observation_columns` in [v2/microgrid_env.py](../v2/microgrid_env.py):
  - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `is_weekend`,
  - `solar_actual_kw`, `solar_forecast_t+1_kw`, `solar_forecast_t+4_kw`,
  - `load_p1_kw`, `load_p2_kw`, `load_p3_kw`,
  - `battery_soc`,
  - `grid_status`, `blackout_probability`, `blackout_type`, `maintenance_active`
- Inference validates these columns in [v2/ppo_inference.py](../v2/ppo_inference.py) and will force `Safe Shutdown` during maintenance.

## Preprocessing
- No normalization/standardization beyond synthetic generation.
- Observations fed directly to PPO; training and inference rely on the above columns’ presence.

## 7-Day Forecast Dataset Notes
- Generated separately by [v2/7.py](../v2/7.py) (not documented here). Must contain the same observation columns as above; includes per-load forecasts and battery characteristics.
- Used by inference to produce [v2/ppo_decisions.csv](../v2/ppo_decisions.csv).

## Caveats
- SOC does not evolve from actions; RL learns policy under exogenous SOC.
- Outage probabilities include rare spikes to promote learning of islanding behavior.
