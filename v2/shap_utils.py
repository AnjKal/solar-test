import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


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
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(x_tensor)

            # Ensure logits are (batch, action_dim)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            # Mirror environment masking rule: when solar_actual_kw == 0, action 1 (charge)
            # must not be selectable.
            try:
                solar_idx = self.observation_columns.index("solar_actual_kw")
            except ValueError:
                solar_idx = None

            if solar_idx is not None and logits.shape[1] > 1:
                solar_kw = x_tensor[:, solar_idx]
                zero_solar = solar_kw <= 0.0
                if bool(torch.any(zero_solar)):
                    logits = logits.clone()
                    logits[zero_solar, 1] = -1e9

            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().numpy()

    def _select_action_shap_values(self, shap_values, action_id: int) -> np.ndarray:
        """Return a 1D array of SHAP contributions (len = n_features) for one sample/output."""

        values = shap_values.values
        if values.ndim != 3:
            raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

        n_features = len(self.observation_columns)

        # Common layouts:
        # - (samples, features, outputs)
        # - (outputs, samples, features)
        if values.shape[0] == 1 and values.shape[1] == n_features:
            # (samples, features, outputs)
            return values[0, :, action_id]

        if values.shape[2] == n_features:
            # (outputs, samples, features)
            return values[action_id, 0, :]

        raise ValueError(
            "Unsupported SHAP axis ordering; got values shape "
            f"{values.shape} with n_features={n_features}"
        )

    def _select_action_explanation(self, shap_values, action_id: int):
        """Return a SHAP Explanation slice for one sample/output for plotting."""

        values = shap_values.values
        n_features = len(self.observation_columns)

        if values.ndim == 3 and values.shape[0] == 1 and values.shape[1] == n_features:
            # (samples, features, outputs)
            return shap_values[0, :, action_id]

        if values.ndim == 3 and values.shape[2] == n_features:
            # (outputs, samples, features)
            return shap_values[action_id, 0, :]

        # Fallback: try the most common multi-output slicing
        try:
            return shap_values[..., action_id][0]
        except Exception as exc:
            raise ValueError(f"Unable to slice SHAP values for action {action_id}") from exc

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
        if self.explainer is None:
            raise RuntimeError("SHAP explainer is not initialized. Call initialize(background_df) first.")

        obs = obs_row[self.observation_columns].values.astype(np.float32)
        obs = obs.reshape(1, -1)

        shap_values = self.explainer(obs)

        # Extract SHAP values for the selected action
        action_shap = self._select_action_shap_values(shap_values, action_id)

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

        action_exp = self._select_action_explanation(shap_values, action_id)
        shap.plots.waterfall(
            action_exp,
            max_display=10,
            show=False
        )

        return fig
