# AI Coding Agent Instructions

## Project Overview
This is a **Reinforcement Learning Microgrid Optimization** project that trains RL agents (PPO and DQN) to optimize energy management in a microgrid system with solar generation, battery storage, and load demand.

**Key Architecture:**
- **Datasets/**: Data generation pipeline using `solarsynth` library for synthetic microgrid data (solar, battery, load)
- **RL_agents/**: Training notebooks for PPO and DQN agents with paper-faithful implementations

## Critical Data Flow
1. **Data Generation** → `Datasets/main.py` calls `solarsynth.synth2` functions to generate CSV files:
   - `solar_actual.csv`, `solar_forecast.csv`, `load_demand.csv`, `battery_soc.csv`
   - Output: `final_microgrid_dataset.csv` (merged dataset with all features)
2. **Training** → PPO/DQN notebooks load the dataset and train agents
3. **Evaluation** → Each agent model is evaluated against the same metrics

## RL Agent Design Patterns

### MicrogridEnv Class (Paper-Faithful)
- **State Space**: `[solar_actual_kw, solar_forecast_kw, battery_soc, load_kw]` (4D)
- **Action Space**: 0=Idle, 1=Charge, 2=Discharge (discrete)
- **Reward Formula**: `+1*load_met - 2*grid_used - 0.5*wasted - 0.1` per step
- **Key Parameters**:
  - `battery_step = 20.0` (SOC % units per action, as in paper)
  - `TIMESTEPS_PER_EPISODE = 24` (24-hour episodes/days)
  - `forecast_enabled` & `high_load` flags for ablation studies

### Training Workflow
- **ActorCritic Network**: Shared encoder + separate actor/critic heads (state_dim=4, hidden_dim=128)
- **PPO Training Loop**:
  1. Collect trajectories (one episode = 24 steps or full dataset)
  2. Compute GAE (Generalized Advantage Estimation)
  3. K_EPOCHS (4) of minibatch updates with clipped objectives
  4. Save checkpoints every 10 episodes + final model
  5. Generate training curve & CSV with rewards

### Hyperparameters (Paper-Aligned)
```python
LR = 3e-4, GAMMA = 0.99, CLIP_EPS = 0.2, ENT_COEF = 0.01
K_EPOCHS = 4, MINI_BATCH_SIZE = 64, N_EPISODES = 10
```
**Never change these without noting it diverges from the referenced paper.**

## Project-Specific Conventions

### Notebook Structure (PPO.ipynb example)
1. **Cell 1**: Imports, reproducibility setup (SEED=123 for all RNG)
2. **Cells 2-3**: Hyperparameters & Environment definition
3. **Cells 4-5**: Network & utility functions (GAE computation)
4. **Cell 6**: `ppo_train()` - main training function returning rewards list & model
5. **Cell 7**: `evaluate_policy()` - deterministic evaluation returning metrics dict
6. **Cell 8**: `main()` - orchestrates training three variants (Baseline, NoForecast, HighLoad)

### Model Naming Convention
- **Baseline**: `PPO_Baseline` (forecast=True, high_load=False)
- **NoForecast**: `PPO_NoForecast` (forecast=False, ablates solar forecast feature)
- **HighLoad**: `PPO_HighLoad` (forecast=True, load 20% higher)

### Output Files
- `{name}_final.pt` - Final trained model weights
- `{name}_ep10.pt` - Checkpoint at episode 10
- `{name}_training_rewards.csv` - Episode rewards for plotting
- `{name}_training_curve.png` - Training curve visualization
- `ppo_experiment_results.csv` - Final evaluation metrics (self_sufficiency_pct, grid_use_kwh, battery_throughput, total_reward)

## Device & Reproducibility
- **Auto-detect GPU**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- **Fixed seed for reproducibility**: SEED=123 for numpy, random, torch (critical for comparing ablations)
- Always ensure `torch.manual_seed(SEED)` before creating models

## Key Integration Points
- **Dataset input**: Must have columns: `solar_actual_kw`, `solar_forecast_kw`, `battery_soc`, `load_kw`
- **Model serialization**: Use `torch.save(model.state_dict(), path)` and `model.load_state_dict(torch.load(path))`
- **External dependency**: PyTorch (RL framework), pandas (data), matplotlib (visualization)

## Debugging Tips
- **Training hangs**: Check dataset row count; `TIMESTEPS_PER_EPISODE=24` requires at least 24 rows per episode
- **Reward inconsistency**: Verify `SEED=123` is set before any RNG call; rerun from cell 1
- **Model evaluation mismatch**: Ensure same `forecast` and `high_load` flags used in training and evaluation
- **Memory issues**: Reduce `MINI_BATCH_SIZE` (currently 64) or `N_EPISODES` (currently 10)

## When Extending the Code
1. **New RL algorithm** (e.g., A2C): Mirror the PPO_train() structure; maintain same state/action/reward definitions
2. **New ablations**: Add to `main()` loop with unique `name`, adjust flags (forecast/high_load), follow naming convention
3. **Environment changes**: Modify `MicrogridEnv.step()` reward logic and update docstring; be explicit if deviating from paper
4. **Dataset changes**: Update expected columns in `MicrogridEnv._get_state()` and Datasets CSV generation
