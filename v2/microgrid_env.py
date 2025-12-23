import gym
from gym import spaces
import numpy as np
import pandas as pd


class MicrogridEnv(gym.Env):
    """
    Production-grade Microgrid Environment
    - Discrete PPO
    - Safe RL (action masking enforced internally)
    - Load prioritization in islanded mode
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, csv_path):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.max_steps = len(self.df)
        self.current_step = 0

        # -------------------------
        # Action Space
        # -------------------------
        """
        0: Grid-connected (idle)
        1: Grid + charge battery
        2: Grid + discharge battery
        3: Islanding (conservative discharge)
        4: Islanding (aggressive discharge)
        5: Safe shutdown
        """
        self.action_space = spaces.Discrete(6)

        # -------------------------
        # Observation Space
        # -------------------------
        self.observation_columns = [
            # Time
            "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend",

            # Solar
            "solar_actual_kw",
            "solar_forecast_t+1_kw",
            "solar_forecast_t+4_kw",

            # Loads
            "load_p1_kw", "load_p2_kw", "load_p3_kw",

            # Battery
            "battery_soc",

            # Grid & safety
            "grid_status",
            "blackout_probability",
            "blackout_type",
            "maintenance_active",
        ]

        self.observation_space = spaces.Box(
            low=-10,
            high=20,
            shape=(len(self.observation_columns),),
            dtype=np.float32
        )

    # -------------------------
    # Helpers
    # -------------------------
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([row[col] for col in self.observation_columns], dtype=np.float32)

    def _is_action_allowed(self, action, row):
        # Hard safety constraints
        if row["maintenance_active"] == 1:
            return action in [0, 5]  # grid-only or shutdown

        if row["grid_status"] == 1 and action in [3, 4]:
            return False  # no islanding if grid is up

        return True

    # -------------------------
    # Step
    # -------------------------
    def step(self, action):
        row = self.df.iloc[self.current_step]

        reward = 0.0
        done = False
        info = {}

        # -------- SAFETY OVERRIDE --------
        if not self._is_action_allowed(action, row):
            reward -= 50.0
            action = 0  # force grid idle

        # -------- LOADS --------
        load_p1 = row["load_p1_kw"]
        load_p2 = row["load_p2_kw"]
        load_p3 = row["load_p3_kw"]

        solar = row["solar_actual_kw"]
        battery_soc = row["battery_soc"]

        # -------- ACTION EFFECTS --------
        is_islanded = action in [3, 4]

        available_power = 0.0
        if is_islanded:
            discharge = 2.5 if action == 3 else 5.0
            available_power = solar + discharge

        # -------- LOAD PRIORITIZATION --------
        served_p1 = served_p2 = served_p3 = 0.0

        if is_islanded:
            served_p1 = min(load_p1, available_power)
            available_power -= served_p1

            served_p2 = min(load_p2, max(0, available_power))
            available_power -= served_p2

            served_p3 = min(load_p3, max(0, available_power))

        else:
            # Grid-connected → serve all
            served_p1, served_p2, served_p3 = load_p1, load_p2, load_p3

        # -------- REWARD FUNCTION --------
        # Critical load violation → catastrophic
        if served_p1 < load_p1:
            reward -= 100.0

        # Positive rewards
        reward += 5.0 * (served_p1 / load_p1)
        reward += 2.0 * (served_p2 / load_p2)
        reward += 0.5 * (served_p3 / load_p3)

        # Penalize unnecessary islanding
        if is_islanded and row["grid_status"] == 1:
            reward -= 5.0

        # Battery protection
        if battery_soc < 0.25:
            reward -= 3.0

        # Maintenance violation (should never happen)
        if row["maintenance_active"] == 1 and is_islanded:
            reward -= 200.0

        # -------- STEP FORWARD --------
        self.current_step += 1
        if self.current_step >= self.max_steps - 1:
            done = True

        return self._get_obs(), reward, done, info

    # -------------------------
    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def render(self, mode="human"):
        pass
