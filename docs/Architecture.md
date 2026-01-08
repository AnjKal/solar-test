# Architecture Documentation

Project: Reinforcement Learning Microgrid Optimization (v2)

This document maps the end-to-end system architecture: data, environment, training, inference, scheduler, and app/explainability. It also compares the implementation to the paper-faithful spec in [.github/copilot-instructions.md](../.github/copilot-instructions.md).

## Components

- Dataset generator: [v2/generate_microgrid_dataset.py](../v2/generate_microgrid_dataset.py)
- Environment: [v2/microgrid_env.py](../v2/microgrid_env.py)
- PPO Agent: [v2/ppo_agent.py](../v2/ppo_agent.py)
- Training script: [v2/train_ppo.py](../v2/train_ppo.py)
- Inference script: [v2/ppo_inference.py](../v2/ppo_inference.py)
- Scheduler: [v2/microgrid_scheduler.py](../v2/microgrid_scheduler.py)
- Forecast generator: [v2/7.py](../v2/7.py)
- Streamlit app: [v2/app.py](../v2/app.py)
- SHAP utilities: [v2/shap_utils.py](../v2/shap_utils.py)

## Data Flow

1. Dataset Generation
   - Run [v2/generate_microgrid_dataset.py](../v2/generate_microgrid_dataset.py) to produce [v2/synthetic_microgrid_dataset.csv](../v2/synthetic_microgrid_dataset.csv).
   - Synthetic data includes time encodings, solar signals and forecasts, prioritized loads, battery SOC/capacity/limits, grid/outage metadata, maintenance windows, islanding indicator, and available island power.

2. Training (PPO)
   - [v2/train_ppo.py](../v2/train_ppo.py) constructs `MicrogridEnv` with the synthetic dataset.
   - Collects rollouts (`ROLLOUT_LENGTH=2048`), computes GAE with `λ=0.95`, PPO updates (`epochs=4`, `clip ε=0.2`, `ent_coef=0.01`, `vf_coef=0.5`, `batch_size=256`).
   - Saves final model weights to [v2/models/ppo_microgrid_final.pth](../v2/models/ppo_microgrid_final.pth).

3. Forecast + Inference
   - The scheduler executes [v2/7.py](../v2/7.py) to generate [v2/next_month_7_day_forecast.csv](../v2/next_month_7_day_forecast.csv) (same schema as observations).
   - [v2/ppo_inference.py](../v2/ppo_inference.py) loads the saved PPO model; applies maintenance overrides; produces greedy actions and state values per row; writes [v2/ppo_decisions.csv](../v2/ppo_decisions.csv).

4. Visualization + Explainability
   - [v2/app.py](../v2/app.py) loads the decisions CSV, recomputes greedy actions, and renders a table of decisions.
   - [v2/shap_utils.py](../v2/shap_utils.py) provides `PPOPolicySHAP` to explain action probabilities; renders bar and waterfall plots.

5. Orchestration
   - [v2/microgrid_scheduler.py](../v2/microgrid_scheduler.py) runs the forecast generator and PPO inference every 5 minutes, logging progress and failures.

## Math Components (Summary)

- Reward (per step) combines: critical load service, islanding penalties if grid-up, battery protection, maintenance constraints, risk preparedness bonuses, forecast-aware nudges, SOC safety margins, and action-flapping penalty. See [docs/Code.md](./Code.md) for the full breakdown.

- PPO Objective:
  - Clipped surrogate `L^{CLIP}` with entropy regularization and value loss; GAE for advantages. See [docs/Code.md](./Code.md) for exact formulas.

## Device & Reproducibility

- Device detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` used in agent, inference, and app.
- Reproducibility:
  - Data generation: `np.random.seed(SEED=42)` set in [v2/generate_microgrid_dataset.py](../v2/generate_microgrid_dataset.py).
  - Training: no explicit seeding in v2; stochastic sampling in PPO means runs are not bitwise reproducible by default.

## Artifacts

- Model weights: [v2/models/ppo_microgrid_final.pth](../v2/models/ppo_microgrid_final.pth)
- Decisions: [v2/ppo_decisions.csv](../v2/ppo_decisions.csv)
- Logs: [v2/ppo_inference.log](../v2/ppo_inference.log), [v2/microgrid_scheduler.log](../v2/microgrid_scheduler.log)

## Alignment vs Paper-Faithful Spec
Reference: [.github/copilot-instructions.md](../.github/copilot-instructions.md)

- State space:
  - Paper: 4D `[solar_actual_kw, solar_forecast_kw, battery_soc, load_kw]`.
  - Implementation: 16D with time encodings, multi-horizon solar, per-priority loads, grid/outage/maintenance features.

- Action space:
  - Paper: 3 actions {Idle, Charge, Discharge}.
  - Implementation: 6 actions, adding islanding modes and safe shutdown.

- Reward:
  - Paper: `+1*load_met − 2*grid_used − 0.5*wasted − 0.1`.
  - Implementation: multi-term reward with safety, forecast-aware nudges, maintenance SOC readiness, etc.

- Battery dynamics:
  - Paper: `battery_step = 20.0` SOC units per action.
  - Implementation: SOC is exogenous; environment does not update SOC based on actions.

- Episode length:
  - Paper: `TIMESTEPS_PER_EPISODE=24`.
  - Implementation: episodes span the full dataset; `done=True` at final index.

- Hyperparameters/artifacts:
  - Core PPO hyperparameters align (LR `3e-4`, `γ=0.99`, `ε=0.2`, `ent=0.01`, epochs=4), but minibatch size differs (`256` vs paper’s `64`).
  - v2 training saves only final weights; paper spec mentions checkpoints and training curves.

## Implications of Divergences

- Larger observation space and extended action set allow richer policies, but deviate from the paper’s constraints; performance and behavior may not be directly comparable.
- Exogenous SOC simplifies dynamics but reduces policy’s ability to learn charge/discharge effects.
- Full-dataset episodes change the variance structure vs day-length episodes.
- Absent seeding may affect reproducibility of training outcomes.

## Suggested Next Steps (Optional)

- Add training seeds and logging of metrics/curves for reproducibility and monitoring.
- Consider introducing SOC transition dynamics to couple actions and battery state.
- If paper-faithful replication is required, add a configuration mode to restrict state/action/reward to the 4D/3-action setup and 24-step episodes.
