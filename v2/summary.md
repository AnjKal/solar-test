# Microgrid RL Summary

This document explains:
- The reward/penalty terms in the microgrid RL environment in real electrical terms
- What each dataset column means electrically
- How the training and forecast datasets are generated (patterns/statistics)

---

## 1) Reward and Penalty Terms (Electrical Meaning)

The environment models a microgrid with PV generation, a battery, prioritized loads, and a grid connection that can fail.

### Key actions (what the agent is “operating”)
- **0**: Grid-connected (idle)
- **1**: Grid + charge battery
- **2**: Island (conservative discharge)
- **3**: Island (aggressive discharge)
- **4**: Safe shutdown (shed all load)
- **5**: Grid-connected discharge + export

### Reward/penalty table

| Term | Real-world electrical meaning | What the code does |
|---|---|---|
| Safety override (invalid action) | Represents illegal/unsafe operator commands (interlocks, protection rules). Examples: exporting with grid down, islanding when grid is healthy (unless very high risk), any action during maintenance. | If an action violates constraints: reward **−50**, then force a safe action (shutdown for maintenance, otherwise idle). |
| High blackout risk (`blackout_probability > 0.6`) | Grid reliability is poor / outage likely soon. Operators should proactively transition to island mode to avoid instability and protect loads. | Under high risk and not maintenance: islanding is enforced (action forced to 2/3 based on expected outage duration). A bonus is still applied for islanding and matching the required mode. |
| Outage duration threshold (120 min) | Long outages require conserving battery (serve only critical loads); short outages can be handled more aggressively. | Chooses required island action: conservative (2) if expected outage ≥120 min, else aggressive (3). |
| Solar-surplus charge shaping | When PV is strong and the grid is healthy, a sensible operator action is to absorb energy by charging the battery (if it has SOC headroom), instead of idling/exporting. | If grid up, not maintenance, not high-risk, solar ≥ threshold and SOC not full: action 1 gets a bonus; idle/export get a small penalty. |
| Solar-surplus threshold (current setting) | Practical “PV is meaningfully available” threshold for opportunistic charge/export decisions when grid is healthy. | Uses `solar_actual_kw > 2.0 kW` together with SOC state: if SOC has headroom → prefer charge; if SOC is full → prefer export. |
| Battery near-full export shaping | If PV is available but the battery is near full, exporting is preferable to idling (otherwise PV may be curtailed). | If grid up, not maintenance, not high-risk, solar > 2.0 kW and SOC is near max: action 5 gets a bonus; action 0 gets a small penalty. |
| Battery SOC bounds (`soc_min=0.1`, `soc_max=0.9`) | Prevents deep discharge (damage/aging) and overcharge (safety/aging). | SOC is clipped; max charge/discharge power is constrained by remaining SOC margin. |
| Charge/discharge efficiencies (0.95) | Converter/inverter + battery losses: energy in/out isn’t 100% efficient. | Charging stores `P·t·η`; discharging consumes `(P·t)/η` from stored energy. |
| Islanding overhead SOC drain | Islanded operation has auxiliary loads and conversion overhead (controls, inverter overhead, comms) that consume energy even beyond “useful” discharge to serve loads. | When action is islanding (2 or 3), SOC is additionally reduced by an overhead draw each step (default 2.0 kW × 1h, applied as battery draw with discharge efficiency). |
| Grid-down wrong-mode penalty | If the grid is unavailable, behaving “grid-connected” isn’t physically meaningful; controller should explicitly choose islanding strategies. | If grid is down (not maintenance) and action isn’t island/shutdown: reward **−5**. |
| Load prioritization (P1 > P2 > P3) | Critical loads (P1) must be served first (e.g., controls, medical, comms). P2/P3 can be shed as needed. | In island operation: allocate available power to P1 first, then P2, then P3. |
| Safe shutdown (action 4) | Intentional load shedding to protect equipment during maintenance or unsafe conditions. | If action 4: served loads set to 0. |
| Export reward (action 5) | Feed-in tariff / net metering: exporting excess energy to grid has value. | If grid available and action 5: reward += exported_kWh × 0.10 (computed from excess after serving loads). |
| Critical load violation penalty | Losing critical service is unacceptable in real operations (hard constraint). | If P1 is not fully served: reward **−100**. |
| Load service rewards | Rewards meeting demand with higher weight on critical load. | Adds: `5*(P1 served / P1)` + `2*(P2 served / P2)` + `0.5*(P3 served / P3)`. |
| Unnecessary islanding penalty | Islanding when grid is healthy can add operational complexity, stress, and synchronization costs. | If islanded while grid is up and not high-risk: reward **−5** and additional penalty scaled by `(1 − blackout_probability)`. |
| Low SOC penalty (`SOC < 0.25`) | Low reserve increases blackout risk and can accelerate degradation. | If SOC < 0.25: reward **−3**. |
| Maintenance violation penalty | During maintenance, equipment may be offline; island actions should not occur. | If maintenance is active and islanded: reward **−200** (should be prevented by action constraints). |
| Risk preparedness bonus (`blackout_probability > 0.8` and SOC healthy) | High risk + high SOC means the microgrid has strong reserve to ride-through outages. | Adds a small bonus proportional to SOC above 0.6. |
| Island efficiency bonus | Successfully meeting critical load while islanded indicates good local dispatch. | If islanded and P1 fully served: **+0.3**. |
| SOC band penalty (0.25–0.35 while islanded) | Discourages operating near low reserve margin in island mode. | If islanded and SOC in [0.25,0.35): **−1.5**. |
| Forecast-aware shaping | Encourages charging before forecast low-solar windows; discourages discharge when future solar is weak. | If grid up and `(solar_f1 + solar_f4) < 1.5`: charge gets **+0.5**, discharge actions get **−0.5**. |
| Maintenance SOC incentives | If maintenance implies grid down, higher SOC improves resilience; low SOC increases risk. | If maintenance: SOC < 0.5 → **−2**; SOC ≥ 0.7 → bonus `(SOC − 0.7)`. |
| Action flapping penalty | Frequent island/grid switching can cause transients, stress, and synchronization issues. | If islanding state toggles vs previous step: **−0.5**. |
| Action switching penalty | Rapidly changing control modes (even within grid-connected or within islanded) can cause unnecessary switching losses and controller churn. | If action changes vs previous step: reward **−0.3** (applies in addition to islanding-state flapping penalty). |
| Battery degradation cost (cycling proxy) | Battery aging is strongly related to throughput/cycling; large SOC swings are discouraged to reflect wear cost. | Each step penalizes SOC movement: reward `− |SOC_after − SOC_before| × 3.0`. |

---

## 2) Dataset Columns (Electrical Meaning and Relevance)

### Time features
| Column | Electrical relevance | Notes |
|---|---|---|
| `timestamp` | Human-readable time index | Not part of the observation vector used by PPO. |
| `hour_sin`, `hour_cos` | Captures daily periodicity without a midnight discontinuity | Helps the policy learn day/night PV/load patterns. |
| `day_sin`, `day_cos` | Captures seasonal / long-horizon periodicity | In the 1-day forecast, `day_*` is constant (single-day scenario). |
| `is_weekend` | Loads often differ on weekends | Binary flag. |

### Solar
| Column | Electrical meaning | Used for |
|---|---|---|
| `solar_actual_kw` | PV real power available locally (kW) | Island supply + export calculations. |
| `solar_forecast_t+1_kw` | 1-hour ahead PV forecast (kW) | Forecast-aware reward shaping. |
| `solar_forecast_t+4_kw` | 4-hour ahead PV forecast (kW) | Forecast-aware reward shaping. |

### Loads
| Column | Electrical meaning | Used for |
|---|---|---|
| `load_p1_kw` | Priority-1 critical demand (kW) | Hard penalty if unmet. |
| `load_p2_kw` | Priority-2 demand (kW) | Lower-weight reward term. |
| `load_p3_kw` | Priority-3 demand (kW) | Lowest-weight reward term. |

### Battery
| Column | Electrical meaning | Used for |
|---|---|---|
| `battery_soc` | SOC fraction (0–1) at reset | Env evolves SOC internally thereafter. |
| `battery_capacity_kwh` | Battery energy capacity (kWh) | Converts SOC ↔ kWh for constraints. |
| `battery_max_charge_kw` | Max charge power (kW) | Limits charging action. |
| `battery_max_discharge_kw` | Max discharge power (kW) | Limits discharge/island/export actions. |

### Grid / reliability / safety
| Column | Electrical meaning | Used for |
|---|---|---|
| `grid_status` | 1 = grid available, 0 = grid down | Determines whether grid-connected operation is feasible. |
| `blackout_probability` | Near-term outage risk proxy (0–1) | Drives high-risk shaping and islanding decisions. |
| `blackout_type` | 0=normal, 1=outage/risk, 2=maintenance outage | Type 2 implies maintenance. |
| `expected_outage_duration_min` | Expected outage duration (minutes) | Chooses conservative vs aggressive islanding. |
| `maintenance_active` | Maintenance flag; implies grid down | Hard constraint: only shutdown allowed during maintenance. |

### Derived (not used by PPO observation vector)
| Column | Meaning | Why it exists |
|---|---|---|
| `is_islanded` | Grid down but not maintenance | Diagnostics/analysis. |
| `available_island_power_kw` | `solar_actual_kw + battery_max_discharge_kw` | Upper-bound proxy for island capability. |

---

## 3) How the Datasets Are Generated

There are two generators:

### A) Training dataset: `generate_synthetic_microgrid_dataset` (in `generate_microgrid_dataset.py`)

Purpose: create a full-year (default 365 days × 24 hours) dataset with realistic-ish variation plus enforced stress conditions.

**Patterns / statistics**
- **Time**: loops `days * 24` hours; computes sine/cosine encodings.
- **Solar**: diurnal “daylight” shape using a sine curve from roughly 06:00–18:00, with a random daily peak and mild noise.
- **Load**:
  - Base random loads with an evening bump (18:00–22:00).
  - 10% chance of “overload” scaling (simulates peak demand days).
- **Battery SOC**: uniform random in `[0.10, 0.90]` for coverage.
- **Blackout/maintenance regime (hard requirements enforced)**:
  - Bernoulli draw for **high blackout** with probability `p_high_blackout = 0.35` (so ≥30% rows end up with `blackout_probability > 0.6`).
  - Conditional Bernoulli for **maintenance given high blackout** with probability `p_maintenance_given_high = 0.50`.
  - Maintenance rows are always generated with:
    - `blackout_type = 2`
    - `maintenance_active = 1`
    - `grid_status = 0`
    - high `blackout_probability` sampled from `[0.65, 1.0]`
  - Non-maintenance high-blackout rows (`blackout_type=1`) mix:
    - true outage (`grid_status=0`) and risk-only (`grid_status=1`) cases
  - Implications are enforced after sampling:
    - `blackout_type==2 ⇒ maintenance_active==1 and grid_status==0`
    - `maintenance_active==1 ⇒ grid_status==0`

Output: `synthetic_microgrid_dataset.csv`

### B) Forecast dataset: `generate_stress_evaluation_7_days` (in `7.py`)

Purpose: generate a one-day (24-row) scenario for scheduled inference.

**Patterns / statistics**
- **Time**: 24 hours starting at `start_date`.
- **Solar**: deterministic diurnal curve with fixed peak (6.5 kW).
- **Load**: deterministic base loads with evening bump.
- **SOC**: fixed at 0.85 (env updates SOC internally during action rollout).
- **Blackout/maintenance**: deterministic “hourly regimes”:
  - 8/24 hours are high blackout (`blackout_probability > 0.6`) ⇒ **33.33%**.
  - 4/24 hours are maintenance (`blackout_type=2`, `maintenance_active=1`, `grid_status=0`) ⇒ **16.67%**.
  - Implications enforced exactly as in training.

Output: `next_month_7_day_forecast.csv` (name retained for scheduler compatibility)
