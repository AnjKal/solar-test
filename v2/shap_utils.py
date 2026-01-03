import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from torch.distributions import Categorical


class PPOPolicySHAP:
    """
    SHAP Explainer for PPO Actor-Critic Policy Network
    Explains: Why action A was chosen over others
    """

    def __init__(
        self,
        model,
        observation_columns,
        device="cpu",
        background_samples=100
    ):
        """
        model: ActorCritic model
        observation_columns: list of feature names
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.observation_columns = observation_columns
        self.background_samples = background_samples

        self.explainer = None

    # -------------------------------------------------
    # Internal forward wrapper for SHAP
    # -------------------------------------------------
    def _policy_forward(self, x):
        """
        x: numpy array (batch_size, obs_dim)
        returns: numpy array (batch_size, action_dim)
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    # -------------------------------------------------
    # Initialize SHAP Explainer
    # -------------------------------------------------
    def initialize(self, background_df):
        """
        background_df: pandas DataFrame (same columns as observation space)
        """
        background = background_df[
            self.observation_columns
        ].values.astype(np.float32)

        if len(background) > self.background_samples:
            background = shap.sample(background, self.background_samples)

        self.explainer = shap.Explainer(
            self._policy_forward,
            background,
            feature_names=self.observation_columns
        )

    # -------------------------------------------------
    # Explain a single state
    # -------------------------------------------------
    def explain_state(self, obs_row, action_id):
        """
        obs_row: pandas Series (single row)
        action_id: int (chosen PPO action)
        """
        obs = obs_row[self.observation_columns].values.astype(np.float32)
        obs = obs.reshape(1, -1)

        shap_values = self.explainer(obs)

        # Extract SHAP values for the selected action
        action_shap = shap_values.values[0, :, action_id]

        explanation_df = pd.DataFrame({
            "Feature": self.observation_columns,
            "SHAP Contribution": action_shap
        }).sort_values(
            by="SHAP Contribution",
            key=abs,
            ascending=False
        )

        return explanation_df, shap_values

    # -------------------------------------------------
    # Plot SHAP bar explanation (Streamlit-ready)
    # -------------------------------------------------
    def plot_bar(self, explanation_df, top_k=10):
        fig, ax = plt.subplots(figsize=(8, 4))

        data = explanation_df.head(top_k)[::-1]

        ax.barh(
            data["Feature"],
            data["SHAP Contribution"]
        )

        ax.set_title("Top Feature Contributions (Policy Decision)")
        ax.set_xlabel("Impact on Action Probability")

        return fig

    # -------------------------------------------------
    # Plot SHAP waterfall (decision logic)
    # -------------------------------------------------
    def plot_waterfall(self, shap_values, action_id):
        fig = plt.figure(figsize=(8, 4))

        shap.plots.waterfall(
            shap_values[0, :, action_id],
            max_display=10,
            show=False
        )

        return fig
