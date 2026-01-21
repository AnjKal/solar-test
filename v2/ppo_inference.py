"""Run PPO policy inference over the latest forecast dataset.

This script reads next_month_7_day_forecast.csv, loads the trained PPO model,
computes greedy actions for each state, and persists the decisions to
the ppo_decisions.csv file for downstream visualization.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from microgrid_env import MicrogridEnv
from ppo_agent import ActorCritic


BASE_DIR = Path(__file__).resolve().parent
FORECAST_PATH = BASE_DIR / "next_month_7_day_forecast.csv"
MODEL_PATH = BASE_DIR / "models" / "ppo_microgrid_final.pth"
OUTPUT_PATH = BASE_DIR / "ppo_decisions.csv"
LOG_PATH = BASE_DIR / "ppo_inference.log"

ACTION_MAP = {
    0: "Grid-connected (Idle)",
    1: "Grid + Charge Battery",
    2: "Island Mode (Conservative)",
    3: "Island Mode (Aggressive)",
    4: "Safe Shutdown",
    5: "Grid + Discharge (Export)",
}


def _override_action(env: MicrogridEnv, row: pd.Series, proposed_action: int) -> int:
    """Apply environment safety constraints so logged decisions match actual allowed behavior."""

    # Maintenance must force shutdown.
    if pd.notna(row.get("maintenance_active", 0)) and int(row.get("maintenance_active", 0)) == 1:
        return 4

    # If blackout probability is high, enforce islanding (mode depends on outage duration).
    if float(row.get("blackout_probability", 0.0)) > 0.6:
        outage_min = float(row.get("expected_outage_duration_min", 0.0))
        return 2 if outage_min >= 120.0 else 3

    # If grid is down, pick an explicit islanding action (to avoid confusing labels like "Grid-connected").
    if pd.notna(row.get("grid_status", 1)) and int(row.get("grid_status", 1)) == 0:
        outage_min = float(row.get("expected_outage_duration_min", 0.0))
        return 2 if outage_min >= 120.0 else 3

    # Otherwise, use the proposed action if allowed, else fall back to idle.
    return proposed_action if env._is_action_allowed(proposed_action, row) else 0

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_inference() -> None:
    logging.info("PPO inference run started")

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found at {FORECAST_PATH}. Run 7.py first."
        )

    forecast_df = pd.read_csv(FORECAST_PATH)
    if forecast_df.empty:
        raise ValueError("Forecast dataset is empty; cannot perform inference.")

    env = MicrogridEnv(str(FORECAST_PATH))
    obs_columns = env.observation_columns
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    missing = set(obs_columns) - set(forecast_df.columns)
    if missing:
        raise ValueError(f"Forecast dataset missing required columns: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, action_dim).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Model weights are incompatible with the current environment action space. "
            "If you recently changed MicrogridEnv.action_space, "
            "retrain the PPO model using train_ppo.py and then rerun inference."
        ) from exc
    model.eval()

    action_ids = []
    action_labels = []
    state_values = []
    internal_battery_socs = []

    # Important: step the environment forward so internal battery SOC evolves and is
    # used in subsequent observations (instead of relying on the CSV's static SOC).
    obs = env.reset()

    for idx in range(len(forecast_df)):
        row = forecast_df.iloc[idx]

        # SOC used by the policy at this timestep (internal env state, not the CSV value).
        internal_battery_socs.append(float(env.battery_soc))

        maintenance_flag = row.get("maintenance_active", 0)
        if pd.notna(maintenance_flag) and int(maintenance_flag) == 1:
            proposed_action = 4
            action = _override_action(env, row, proposed_action)
            action_ids.append(action)
            action_labels.append(ACTION_MAP[action])
            state_values.append(np.nan)
            logging.info(
                "Maintenance override at row %s; forcing Safe Shutdown without model inference",
                idx,
            )
            obs, reward, done, info = env.step(action)
            if done:
                break
            continue

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action_mask = env.get_action_mask()
        mask_t = torch.tensor(action_mask, dtype=torch.bool).to(device)

        with torch.no_grad():
            logits, value = model(obs_tensor)
            logits = logits.masked_fill(~mask_t, -1e9)
            dist = Categorical(logits=logits)
            proposed_action = int(torch.argmax(dist.probs).item())

        action = _override_action(env, row, proposed_action)
        action_ids.append(action)
        action_labels.append(ACTION_MAP[action])
        state_values.append(round(value.item(), 3))

        obs, reward, done, info = env.step(action)
        if done:
            break

    results_df = forecast_df.iloc[: len(action_ids)].copy()

    # Preserve the dataset SOC for debugging/traceability, but make battery_soc reflect
    # the internal SOC trajectory actually used during inference.
    if "battery_soc" in results_df.columns:
        results_df["battery_soc_dataset"] = results_df["battery_soc"]
    results_df["battery_soc"] = internal_battery_socs[: len(action_ids)]

    results_df["action_id"] = action_ids
    results_df["action_label"] = action_labels
    results_df["state_value"] = state_values

    results_df.to_csv(OUTPUT_PATH, index=False)

    logging.info("PPO inference completed successfully")


if __name__ == "__main__":
    run_inference()
