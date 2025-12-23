import numpy as np
import pandas as pd
import shap
import torch
from stable_baselines3 import PPO
from microgrid_env import MicrogridEnv

# -----------------------------
# Load model & environment
# -----------------------------
MODEL_PATH = "models/ppo_microgrid_final.zip"
DATASET_PATH = "synthetic_microgrid_dataset.csv"

env = MicrogridEnv(DATASET_PATH)
model = PPO.load(MODEL_PATH)

policy = model.policy
policy.eval()

# -----------------------------
# Observation columns (must match env)
# -----------------------------
feature_names = env.observation_columns

# -----------------------------
# Wrapper for SHAP
# -----------------------------
class PPOPolicyWrapper:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, x):
        """
        x: numpy array [N, features]
        returns: action logits
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            latent_pi, _ = self.policy.mlp_extractor(x_tensor)
            logits = self.policy.action_net(latent_pi)
        return logits.numpy()

wrapper = PPOPolicyWrapper(policy)

# -----------------------------
# Background & test samples
# -----------------------------
df = pd.read_csv(DATASET_PATH)
X = df[feature_names].values

background = X[np.random.choice(len(X), 200, replace=False)]
test_samples = X[np.random.choice(len(X), 5, replace=False)]

# -----------------------------
# SHAP explainer
# -----------------------------
explainer = shap.DeepExplainer(wrapper, background)
shap_values = explainer.shap_values(test_samples)

# -----------------------------
# Save SHAP values
# -----------------------------
np.save("shap_values.npy", shap_values)
np.save("shap_samples.npy", test_samples)

print("✅ SHAP values generated and saved.")
