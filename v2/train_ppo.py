import os
import numpy as np
import torch
from microgrid_env import MicrogridEnv
from ppo_agent import PPOAgent
DATASET_PATH = "synthetic_microgrid_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

env = MicrogridEnv(DATASET_PATH)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(obs_dim, action_dim)
TIMESTEPS = 500_000
ROLLOUT_LENGTH = 2048

obs = env.reset()

buffer = {
    "obs": [],
    "actions": [],
    "log_probs": [],
    "rewards": [],
    "dones": [],
    "values": [],
    "action_masks": [],
}

step = 0

while step < TIMESTEPS:
    buffer = {k: [] for k in buffer}
    
    for _ in range(ROLLOUT_LENGTH):
        action_mask = env.get_action_mask()
        action, log_prob, value = agent.select_action(obs, action_mask=action_mask)
        next_obs, reward, done, _ = env.step(action)

        buffer["obs"].append(obs)
        buffer["actions"].append(action)
        buffer["log_probs"].append(log_prob)
        buffer["rewards"].append(reward)
        buffer["dones"].append(done)
        buffer["values"].append(value.item())
        buffer["action_masks"].append(action_mask)

        obs = next_obs
        step += 1

        if done:
            obs = env.reset()

    with torch.no_grad():
        _, next_value = agent.model(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        )
        next_value = next_value.item()

    advantages = agent.compute_gae(
        buffer["rewards"],
        buffer["values"],
        buffer["dones"],
        next_value
    )

    returns = np.array(advantages) + np.array(buffer["values"])

    metrics = agent.update(
        buffer["obs"],
        buffer["actions"],
        buffer["log_probs"],
        returns,
        advantages,
        action_masks=buffer["action_masks"],
    )

    rewards_total = float(np.sum(buffer["rewards"]))
    print(
        "Training step: "
        f"{step} | rewards_total={rewards_total:.3f} | "
        f"kl_div={metrics['kl_div']:.6f} | policy_entropy={metrics['policy_entropy']:.6f} | "
        f"policy_loss={metrics['policy_loss']:.6f} | value_loss={metrics['value_loss']:.6f} | "
        f"entropy_loss={metrics['entropy_loss']:.6f}"
    )
torch.save(
    agent.model.state_dict(),
    os.path.join(MODEL_DIR, "ppo_microgrid_final.pth")
)

print("✅ PPO training completed and model saved.")
