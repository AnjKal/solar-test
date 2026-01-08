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
    2: "Grid + Discharge Battery",
    3: "Island Mode (Conservative)",
    4: "Island Mode (Aggressive)",
    5: "Safe Shutdown",
    6: "Grid + Discharge (Export)",
}

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
            "If you recently changed MicrogridEnv.action_space (e.g., from 6 to 7 actions), "
            "retrain the PPO model using train_ppo.py and then rerun inference."
        ) from exc
    model.eval()

    action_ids = []
    action_labels = []
    state_values = []

    for idx, row in forecast_df.iterrows():
        maintenance_flag = row["maintenance_active"]
        if pd.notna(maintenance_flag) and int(maintenance_flag) == 1:
            action = 5
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
