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

    for idx, row in forecast_df.iterrows():
        maintenance_flag = row["maintenance_active"]
        if pd.notna(maintenance_flag) and int(maintenance_flag) == 1:
            action = 4
            action_ids.append(action)
            action_labels.append(ACTION_MAP[action])
            state_values.append(np.nan)
            logging.info(
                "Maintenance override at row %s; forcing Safe Shutdown without model inference",
                idx,
            )
            continue

        obs = row[obs_columns].values.astype(np.float32)
        obs_tensor = torch.tensor(obs).to(device)

        with torch.no_grad():
            logits, value = model(obs_tensor)
            dist = Categorical(logits=logits)
            action = torch.argmax(dist.probs).item()

        action = _override_action(env, row, int(action))

        action_ids.append(action)
        action_labels.append(ACTION_MAP[action])
        state_values.append(round(value.item(), 3))

    results_df = forecast_df.copy()
    results_df["action_id"] = action_ids
    results_df["action_label"] = action_labels
    results_df["state_value"] = state_values

    results_df.to_csv(OUTPUT_PATH, index=False)

    logging.info("PPO inference completed successfully")


if __name__ == "__main__":
    run_inference()
