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
        self.prev_islanded = None

        # Battery SOC is modeled as an internal state that evolves with actions.
        # The dataset SOC (if present) is used only to initialize/reset.
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        self.timestep_hours = 1.0

        initial_soc = float(self.df.iloc[0].get("battery_soc", 0.6))
        self.initial_battery_soc = float(np.clip(initial_soc, self.soc_min, self.soc_max))
        self.battery_soc = self.initial_battery_soc

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
        6: Grid-connected battery discharge with grid export
        """
        self.action_space = spaces.Discrete(7)

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
        obs = []
        for col in self.observation_columns:
            if col == "battery_soc":
                obs.append(self.battery_soc)
            else:
                obs.append(row[col])
        return np.array(obs, dtype=np.float32)

    def _is_action_allowed(self, action, row):
        # Hard safety constraints
        if row["maintenance_active"] == 1:
            # Maintenance implies grid is down; only safe shutdown is allowed.
            return action == 5

        # Normally: no islanding if grid is up.
        # Exception: if blackout risk is high, preemptive islanding is allowed.
        if row["grid_status"] == 1 and action in [3, 4]:
            return float(row.get("blackout_probability", 0.0)) > 0.6

        # Export action requires grid availability.
        if action == 6:
            return row["grid_status"] == 1

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
            action = 5 if int(row["maintenance_active"]) == 1 else 0

        # -------- BLACKOUT-RISK POLICY SHAPING --------
        # If blackout probability is high, the agent should proactively island.
        # Mode selection depends on expected outage duration:
        # - long outage  -> conservative (preserve battery)
        # - short outage -> aggressive (serve more load)
        bp = float(row["blackout_probability"])
        outage_min = float(row.get("expected_outage_duration_min", 0.0))
        high_risk = (bp > 0.6) and (int(row["maintenance_active"]) != 1)
        if high_risk:
            long_outage_threshold_min = 120.0
            required_island_action = 3 if outage_min >= long_outage_threshold_min else 4

            if action not in [3, 4]:
                # Strong penalty for not islanding under high risk.
                reward -= 10.0
            else:
                # Small bonus for islanding under high risk.
                reward += 2.0
                if action == required_island_action:
                    reward += 2.0
                else:
                    reward -= 2.0

        # -------- BATTERY DYNAMICS (DYNAMIC SOC) --------
        # Charging: action 1
        # Discharging: actions 2, 3, 4, 6
        battery_capacity_kwh = float(row.get("battery_capacity_kwh", 20.0))
        battery_max_charge_kw = float(row.get("battery_max_charge_kw", 5.0))
        battery_max_discharge_kw = float(row.get("battery_max_discharge_kw", 5.0))

        target_charge_kw = 0.0
        target_discharge_kw = 0.0

        if action == 1:
            target_charge_kw = battery_max_charge_kw
        elif action in [2, 6]:
            target_discharge_kw = battery_max_discharge_kw
        elif action == 3:
            target_discharge_kw = min(2.5, battery_max_discharge_kw)
        elif action == 4:
            target_discharge_kw = min(5.0, battery_max_discharge_kw)

        # SOC-constrained charge
        charge_kw = 0.0
        if target_charge_kw > 0.0:
            max_charge_kw_soc = ((self.soc_max - self.battery_soc) * battery_capacity_kwh) / (
                self.timestep_hours * self.charge_eff
            )
            charge_kw = float(np.clip(target_charge_kw, 0.0, min(battery_max_charge_kw, max_charge_kw_soc)))

        # SOC-constrained discharge
        discharge_kw = 0.0
        if target_discharge_kw > 0.0:
            deliverable_kwh = (self.battery_soc - self.soc_min) * battery_capacity_kwh * self.discharge_eff
            max_discharge_kw_soc = deliverable_kwh / self.timestep_hours
            discharge_kw = float(np.clip(target_discharge_kw, 0.0, min(battery_max_discharge_kw, max_discharge_kw_soc)))

        # Update SOC (apply efficiency losses)
        if charge_kw > 0.0 and discharge_kw > 0.0:
            # Should not happen with the current action set, but keep conservative behavior.
            if charge_kw >= discharge_kw:
                discharge_kw = 0.0
            else:
                charge_kw = 0.0

        if charge_kw > 0.0:
            charged_kwh = charge_kw * self.timestep_hours * self.charge_eff
            self.battery_soc = float(np.clip(self.battery_soc + (charged_kwh / battery_capacity_kwh), self.soc_min, self.soc_max))

        if discharge_kw > 0.0:
            battery_draw_kwh = (discharge_kw * self.timestep_hours) / self.discharge_eff
            self.battery_soc = float(np.clip(self.battery_soc - (battery_draw_kwh / battery_capacity_kwh), self.soc_min, self.soc_max))

        # -------- LOADS --------
        load_p1 = row["load_p1_kw"]
        load_p2 = row["load_p2_kw"]
        load_p3 = row["load_p3_kw"]

        solar = row["solar_actual_kw"]
        battery_soc = self.battery_soc

        # -------- ACTION EFFECTS --------
        is_islanded = action in [3, 4]

        # Action 6 is grid-connected export mode (not islanded)
        is_export = action == 6

        grid_available = (int(row["grid_status"]) == 1) and (int(row["maintenance_active"]) != 1)
        effective_islanded = is_islanded or ((not grid_available) and action != 5)

        # If the grid is down and we're not explicitly islanding, penalize.
        # (The system will still operate islanded in practice, but we want the policy
        # to choose the appropriate island-mode actions.)
        if (not grid_available) and int(row["maintenance_active"]) != 1 and (action not in [3, 4, 5]):
            reward -= 5.0

        available_power = 0.0
        if effective_islanded:
            available_power = solar + discharge_kw

        # -------- LOAD PRIORITIZATION --------
        served_p1 = served_p2 = served_p3 = 0.0

        if action == 5:
            # Safe shutdown
            served_p1 = served_p2 = served_p3 = 0.0
        elif effective_islanded:
            served_p1 = min(load_p1, available_power)
            available_power -= served_p1

            served_p2 = min(load_p2, max(0, available_power))
            available_power -= served_p2

            served_p3 = min(load_p3, max(0, available_power))

        else:
            # Grid-connected and grid available → serve all
            served_p1, served_p2, served_p3 = load_p1, load_p2, load_p3

        # -------- GRID EXPORT REWARD (ACTION 6) --------
        # When grid is available and action 6 is chosen, export any excess power
        # after serving local loads. Local loads are always fully served in grid-connected mode.
        if is_export and row["grid_status"] == 1:
            total_load_kw = float(load_p1 + load_p2 + load_p3)
            total_available_kw = float(solar + discharge_kw)
            excess_kw = max(0.0, total_available_kw - total_load_kw)

            export_kwh = excess_kw * self.timestep_hours
            feed_in_tariff = 0.10  # reward units per kWh exported (tunable)
            reward += export_kwh * feed_in_tariff

        # -------- REWARD FUNCTION --------
        # Critical load violation → catastrophic
        if served_p1 < load_p1:
            reward -= 100.0

        # Positive rewards
        reward += 5.0 * (served_p1 / load_p1)
        reward += 2.0 * (served_p2 / load_p2)
        reward += 0.5 * (served_p3 / load_p3)

        # Penalize unnecessary islanding (when blackout risk is not high)
        if is_islanded and row["grid_status"] == 1 and not high_risk:
            reward -= 5.0
            reward -= 2.0 * (1.0 - float(row["blackout_probability"]))

        # Battery protection
        if battery_soc < 0.25:
            reward -= 3.0

        # Maintenance violation (should never happen)
        if row["maintenance_active"] == 1 and is_islanded:
            reward -= 200.0

        # Risk preparedness bonus when blackout risk is high and SOC is healthy
        if float(row["blackout_probability"]) > 0.8 and battery_soc >= 0.6:
            reward += max(0.0, float(battery_soc) - 0.6)

        # Island efficiency bonus for meeting critical load while islanded
        if is_islanded and served_p1 >= load_p1:
            reward += 0.3

        # SOC safety margin penalty while islanded (soft band)
        if is_islanded and 0.25 <= float(battery_soc) < 0.35:
            reward -= 1.5

        # Forecast-aware behavior: encourage charging before low solar, discourage discharging
        low_forecast = (float(row["solar_forecast_t+1_kw"]) + float(row["solar_forecast_t+4_kw"])) < 1.5
        if row["grid_status"] == 1 and low_forecast:
            if action == 1:
                reward += 0.5
            elif action in [2, 4]:
                reward -= 0.5

        # Maintenance SOC readiness incentives
        if row["maintenance_active"] == 1:
            if float(battery_soc) < 0.5:
                reward -= 2.0
            elif float(battery_soc) >= 0.7:
                reward += (float(battery_soc) - 0.7)

        # Action flapping penalty (toggle between islanded and grid-connected)
        if self.prev_islanded is not None and bool(self.prev_islanded) != bool(is_islanded):
            reward -= 0.5

        # -------- STEP FORWARD --------
        self.current_step += 1
        self.prev_islanded = is_islanded
        if self.current_step >= self.max_steps - 1:
            done = True

        return self._get_obs(), reward, done, info

    # -------------------------
    def reset(self):
        self.current_step = 0
        self.prev_islanded = None
        return self._get_obs()

    def render(self, mode="human"):
        pass
