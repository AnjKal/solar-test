# Code Documentation

Project: Reinforcement Learning Microgrid Optimization (v2)

This document describes the microgrid environment logic, PPO agent architecture and training loop, and inference behavior, with formal math where applicable.

## MicrogridEnv
Source: [v2/microgrid_env.py](../v2/microgrid_env.py)

- Observation space (`obs_dim=16`):
  - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `is_weekend`
  - `solar_actual_kw`, `solar_forecast_t+1_kw`, `solar_forecast_t+4_kw`
  - `load_p1_kw`, `load_p2_kw`, `load_p3_kw`
  - `battery_soc`
  - `grid_status`, `blackout_probability`, `blackout_type`, `maintenance_active`
  - Box bounds defined as `low=-10`, `high=20` for all features (float32).

- Action space (`Discrete(6)`):
  - `0`: Grid-connected (idle)
  - `1`: Grid + charge battery
  - `2`: Grid + discharge battery
  - `3`: Islanding (conservative discharge)
  - `4`: Islanding (aggressive discharge)
  - `5`: Safe shutdown
  - `6`: Grid-connected discharge (export)

- Safety/action masking:
  - If `maintenance_active==1`: only `{0,5}` permitted.
  - If `grid_status==1` (grid up): island actions `{3,4}` disallowed.
  - Disallowed actions are overridden to `0` (idle) and incur `−50` penalty.

- Islanding available power:
  - Discharge power is SOC- and limit-constrained.
  - Target discharge is `2.5 kW` (action `3`) or `5.0 kW` (action `4`) and is clamped by `battery_max_discharge_kw` and SOC.
  - `available_power = solar_actual_kw + discharge_kw`.

- Grid export (action `6`):
  - When `grid_status==1`, compute `excess_kw = max(0, solar_actual_kw + discharge_kw − (load_p1_kw+load_p2_kw+load_p3_kw))`.
  - Reward export using a feed-in tariff: `reward += export_kwh × tariff`, where `export_kwh = excess_kw × timestep_hours`.

- Load prioritization (islanded only): serve `p1 → p2 → p3` greedily from `available_power`.
  - Grid-connected: all loads served.

- Episode and reset:
  - `max_steps = len(dataset)`; `done=True` at final index.
  - `reset()` sets `current_step=0`, clears `prev_islanded`.

### Reward Function (per step)
Let `L_i` be loads and `S_i` be served loads for `i ∈ {1,2,3}`; `I` indicates islanding; `G` grid up flag; `bp` blackout probability; `soc` battery SOC.

- Catastrophic critical-load miss: if `S_1 < L_1`, add `−100`.
- Load service terms:
  - `+ 5 · (S_1 / L_1) + 2 · (S_2 / L_2) + 0.5 · (S_3 / L_3)`.
- Unnecessary islanding penalties (grid up):
  - if `I=1` and `G=1`, add `−5 − 2 · (1 − bp)`.
- Battery protection: if `soc < 0.25`, add `−3`.
- Maintenance violation: if `maintenance_active=1` and `I=1`, add `−200`.
- Risk preparedness bonus: if `bp > 0.8` and `soc ≥ 0.6`, add `max(0, soc − 0.6)`.
- Island efficiency: if `I=1` and `S_1 ≥ L_1`, add `+0.3`.
- SOC safety margin: if `I=1` and `0.25 ≤ soc < 0.35`, add `−1.5`.
- Forecast-aware behavior (grid up and low solar ahead):
  - define `low_forecast = (solar_forecast_t+1_kw + solar_forecast_t+4_kw) < 1.5`.
  - if `low_forecast` and action==`1` (charge), add `+0.5`; if action in `{2,4}`, add `−0.5`.
- Maintenance SOC readiness:
  - if `maintenance_active=1` and `soc < 0.5`, add `−2`.
  - if `maintenance_active=1` and `soc ≥ 0.7`, add `soc − 0.7`.
- Action flapping penalty: if islanding toggles since last step, add `−0.5`.

The total reward is the sum of the above terms.

## PPO Agent
Source: [v2/ppo_agent.py](../v2/ppo_agent.py)

- Actor-Critic network (`ActorCritic`):
  - Shared MLP: `Linear(obs_dim→128) + ReLU + Linear(128→128) + ReLU`.
  - Policy head: `Linear(128→action_dim)` logits.
  - Value head: `Linear(128→1)` scalar state value.

- Device: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

- Hyperparameters:
  - `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`,
  - `vf_coef=0.5` (value loss weight), `ent_coef=0.01` (entropy bonus).
  - Gradient norm clip: `0.5`.

- Action selection:
  - `Categorical(logits=policy)`; training samples stochastically (`dist.sample()`).

- Advantage (GAE):
  - For rewards `r_t`, values `V_t`, dones `d_t`, bootstrap `V_{T}`:
  - `δ_t = r_t + γ · V_{t+1} · (1 − d_t) − V_t`
  - `GAE_t = δ_t + γ · λ · (1 − d_t) · GAE_{t+1}` (backward recursion)
  - Returns `R_t = GAE_t + V_t`.

- PPO clipped objective:
  - Ratio `r_t(θ) = exp(log π_θ(a_t|s_t) − log π_{θ_old}(a_t|s_t))`.
  - Surrogate `L^{CLIP} = E[min(r_t · A_t, clip(r_t, 1−ε, 1+ε) · A_t)]`.
  - Value loss `L^V = MSE(V_θ(s_t), R_t)`; entropy `H(π_θ)`.
  - Total loss `L = −L^{CLIP} + c_v · L^V − c_e · H`.

- Update (`epochs=4`, `batch_size=256`):
  - Shuffle indices; compute losses per minibatch; backprop and step.

## Training Loop
Source: [v2/train_ppo.py](../v2/train_ppo.py)

- Environment: [v2/microgrid_env.py](../v2/microgrid_env.py) with dataset [v2/synthetic_microgrid_dataset.csv](../v2/synthetic_microgrid_dataset.csv).
- Constants: `TIMESTEPS=500_000`, `ROLLOUT_LENGTH=2048`.
- Buffer collects `obs, actions, log_probs, rewards, dones, values`.
- Rollout until `ROLLOUT_LENGTH`; reset environment when `done`.
- Bootstrap next value; compute `advantages` via GAE; `returns = advantages + values`.
- Call `agent.update(...)` with default PPO hyperparameters.
- Save final weights: [v2/models/ppo_microgrid_final.pth](../v2/models/ppo_microgrid_final.pth).

## Inference
Source: [v2/ppo_inference.py](../v2/ppo_inference.py)

- Input: [v2/next_month_7_day_forecast.csv](../v2/next_month_7_day_forecast.csv).
- Validates required observation columns (same as env); initializes `ActorCritic` with saved weights.
- Maintenance override: if `maintenance_active==1`, set action to `5` (Safe Shutdown) without model evaluation.
- Otherwise, greedy action selection (`argmax` over action probabilities); also logs `state_value`.
- Output: [v2/ppo_decisions.csv](../v2/ppo_decisions.csv).

## Streamlit App
Source: [v2/app.py](../v2/app.py)

- Loads decisions CSV; recomputes greedy actions and state values for display.
- Integrates SHAP via [v2/shap_utils.py](../v2/shap_utils.py) to explain policy decisions:
  - `PPOPolicySHAP` wraps the ActorCritic policy; explains action probability contributions per feature.
  - Provides bar and waterfall plots for top contributors.

## Notes and Caveats
- SOC is modeled dynamically inside the environment and evolves based on actions (charge/discharge), battery limits, efficiencies, timestep duration, and SOC bounds.
- Islanding discharge levels are fixed (2.5 kW, 5.0 kW) rather than battery-step dynamics.
- Training loop saves only final weights; no checkpoints/curves produced by default in v2.
