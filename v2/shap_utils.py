import torch
import shap
import numpy as np


class PPOPolicyWrapper:
    """
    Wraps PPO ActorCritic to expose a single action logit for SHAP
    """

    def __init__(self, model, device, action_index):
        self.model = model
        self.device = device
        self.action_index = action_index

    def __call__(self, x):
        """
        x: numpy array of shape (N, obs_dim)
        returns: numpy array (N,) → logit for selected action
        """
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(x)

        return logits[:, self.action_index].cpu().numpy()


def get_shap_explainer(model, device, background, action_idx):
    """
    Creates a SHAP KernelExplainer for PPO policy logits
    """
    wrapped_policy = PPOPolicyWrapper(
        model=model,
        device=device,
        action_index=action_idx
    )

    explainer = shap.KernelExplainer(
        wrapped_policy,
        background
    )

    return explainer
