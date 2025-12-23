import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch.distributions import Categorical
import shap

from shap_utils import PPOPolicyWrapper, get_shap_explainer
from microgrid_env import MicrogridEnv
from ppo_agent import ActorCritic


# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Microgrid EMS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #020617;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}
.glass {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 20px;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}
.subtle {
    color: #94a3b8;
    font-size: 14px;
}
button {
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA + MODEL
# ============================================================
DATASET_PATH = "synthetic_microgrid_dataset.csv"
MODEL_PATH = "models/ppo_microgrid_final.pth"

df = pd.read_csv(DATASET_PATH)
env = MicrogridEnv(DATASET_PATH)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCritic(obs_dim, action_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ============================================================
# SESSION STATE
# ============================================================
if "maintenance" not in st.session_state:
    st.session_state.maintenance = []


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="glass">
  <h1>⚡ AI-Driven Microgrid Energy Management System</h1>
  <p class="subtle">
    PPO-controlled microgrid with safety constraints and explainability
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAINTENANCE UI
# ============================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🛠 Maintenance Scheduling")

    start_h, end_h = st.slider(
        "Repair Window (Hours)",
        0, 23, (10, 14),
        label_visibility="collapsed"
    )

    if st.button("Add Maintenance Window"):
        st.session_state.maintenance.append((start_h, end_h))

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Active Windows")

    if st.session_state.maintenance:
        for s, e in st.session_state.maintenance:
            st.markdown(f"🟥 **Hour {s} → {e}**")
    else:
        st.markdown("_None_")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# RUN PPO POLICY
# ============================================================
env.reset()
# env.set_maintenance_schedule(st.session_state.maintenance)

actions = []
rewards = []

obs = env.reset()

# SHAP background (ONLY here)
background = df.sample(100).values[:, :obs_dim]

for t in range(24):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    logits, _ = model(obs_tensor)
    dist = Categorical(logits=logits)
    action = dist.probs.argmax().item()

    obs, reward, done, _ = env.step(action)

    actions.append(action)
    rewards.append(reward)


# ============================================================
# ACTION TIMELINE PLOT
# ============================================================
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=actions,
    mode="lines+markers",
    name="PPO Action",
    line=dict(width=4)
))

for s, e in st.session_state.maintenance:
    fig.add_vrect(
        x0=s, x1=e,
        fillcolor="red",
        opacity=0.25,
        layer="below",
        line_width=0
    )

fig.update_layout(
    template="plotly_dark",
    height=420,
    xaxis_title="Hour",
    yaxis_title="Action ID",
    title="PPO Decision Schedule with Safety Overrides"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="glass">
<h3>🔍 Why did the PPO choose this action?</h3>
<p class="subtle">
Local explanation of the PPO policy logits
</p>
</div>
""", unsafe_allow_html=True)

# Explain LAST action
last_action = actions[-1]
current_state = np.array(obs).reshape(1, -1)

explainer = get_shap_explainer(
    model=model,
    device=device,
    background=background,
    action_idx=last_action
)

shap_values = explainer.shap_values(
    current_state,
    nsamples=200
)

feature_names = [
    "Hour (sin)",
    "Hour (cos)",
    "Day (sin)",
    "Day (cos)",
    "Is Weekend",

    "Solar Actual (kW)",
    "Solar Forecast t+1 (kW)",
    "Solar Forecast t+4 (kW)",

    "Load Priority 1 (kW)",
    "Load Priority 2 (kW)",
    "Load Priority 3 (kW)",

    "Battery SOC",

    "Grid Status",
    "Blackout Probability",
    "Blackout Type",

    "Maintenance Active"
]


shap.initjs()

import matplotlib.pyplot as plt

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=current_state[0],
        feature_names=feature_names
    ),
    show=False
)

fig = plt.gcf()   # ⬅️ get current Figure
st.pyplot(fig, bbox_inches="tight")
plt.close(fig)


if df.iloc[env.current_step - 1]["maintenance_active"] == 1:
    st.info(
        "Explanation shown is hypothetical. "
        "Final action overridden due to maintenance safety constraints."
    )
