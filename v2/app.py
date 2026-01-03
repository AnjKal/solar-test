from pathlib import Path

import streamlit as st
import torch
import pandas as pd
import numpy as np
from torch.distributions import Categorical

from microgrid_env import MicrogridEnv
from ppo_agent import ActorCritic
from shap_utils import PPOPolicySHAP

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ppo_microgrid_final.pth"
DECISIONS_PATH = BASE_DIR / "ppo_decisions.csv"

ACTION_MAP = {
    0: "Grid-connected (Idle)",
    1: "Grid + Charge Battery",
    2: "Grid + Discharge Battery",
    3: "Island Mode (Conservative)",
    4: "Island Mode (Aggressive)",
    5: "Safe Shutdown"
}

st.set_page_config(
    page_title="PPO Microgrid Decisions + Explainability",
    layout="wide"
)

# --------------------------------------------------
# LOAD ENV + DATA
# --------------------------------------------------
def load_decisions() -> pd.DataFrame:
    if not DECISIONS_PATH.exists():
        st.error(
            "❌ PPO decisions not found. Start the scheduler via `python microgrid_scheduler.py`."
        )
        st.stop()

    return pd.read_csv(DECISIONS_PATH)

decisions_df = load_decisions()

env = MicrogridEnv(str(DECISIONS_PATH))
obs_columns = env.observation_columns
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

df = decisions_df.copy()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCritic(obs_dim, action_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --------------------------------------------------
# INIT SHAP EXPLAINER
# --------------------------------------------------
shap_explainer = PPOPolicySHAP(
    model=model,
    observation_columns=obs_columns,
    device=device,
)
shap_explainer.initialize(df)

# --------------------------------------------------
# UI HEADER
# --------------------------------------------------
st.title("⚡ PPO Microgrid Decision Viewer with Explainability")
st.markdown(
    "Each row represents a **microgrid state**. "
    "The PPO agent selects an action. "
    "Click **Explain** to see why."
)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
if df.empty:
    st.error("❌ PPO decisions dataset is empty.")
    st.stop()

missing = set(obs_columns) - set(df.columns)
if missing:
    st.error(f"❌ Missing required columns in PPO decisions file: {missing}")
    st.stop()

# --------------------------------------------------
# INFERENCE + DISPLAY
# --------------------------------------------------
st.subheader("🤖 PPO Decisions")

action_ids = []
action_labels = []
state_values = []

for _, row in df.iterrows():
    obs = row[obs_columns].values.astype(np.float32)
    obs_tensor = torch.tensor(obs).to(device)

    with torch.no_grad():
        logits, value = model(obs_tensor)
        dist = Categorical(logits=logits)
        action = torch.argmax(dist.probs).item()

    action_ids.append(action)
    action_labels.append(ACTION_MAP[action])
    state_values.append(round(value.item(), 3))

df["action_id"] = action_ids
df["action_label"] = action_labels
df["state_value"] = state_values

summary_columns = [
    "action_id",
    "action_label",
    "state_value",
] + obs_columns

st.dataframe(df[summary_columns], use_container_width=True)

selected_index = st.selectbox(
    "Select a state for explainability",
    options=df.index,
    format_func=lambda idx: f"Row {idx} • {df.loc[idx, 'action_label']}"
)

if st.button("🧠 Explain Selected Decision"):
    explanation_df, shap_values = shap_explainer.explain_state(
        df.loc[selected_index],
        int(df.loc[selected_index, "action_id"]),
    )

    st.markdown("### 🔍 Decision Explainability")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### 📊 Feature Contributions")
        st.dataframe(explanation_df.head(10), use_container_width=True)

    with colB:
        st.markdown("#### 📈 SHAP Bar Plot")
        fig_bar = shap_explainer.plot_bar(explanation_df)
        st.pyplot(fig_bar)

    st.markdown("#### 🧠 Decision Waterfall")
    fig_waterfall = shap_explainer.plot_waterfall(
        shap_values,
        int(df.loc[selected_index, "action_id"]),
    )
    st.pyplot(fig_waterfall)
