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
    - Cost-aware + degradation-aware
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, csv_path):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.max_steps = len(self.df)
        self.current_step = 0
        self.prev_islanded = None
        self.prev_action = None

        # Battery SOC is modeled as an internal state that evolves with actions.
        # The dataset SOC (if present) is used only to initialize/reset.
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        self.timestep_hours = 1.0

        # Reward shaping thresholds (tunable)
        self.solar_charge_threshold_kw = 2.0
        self.soc_headroom_epsilon = 1e-3
        # High reward for charging when solar surplus + SOC headroom (aligned with night idle reward magnitude)
        self.solar_charge_bonus = 6.0
        self.solar_charge_miss_penalty = 6.0

        # Night-time (no-solar) preferences (tunable)
        # If solar is ~0 and no other constraints apply, prefer:
        # - SOC < threshold -> grid-connected idle (0)
        # - SOC >= threshold -> discharge+export (5)
        self.night_solar_epsilon_kw = 0
        self.night_soc_export_threshold = 0.5
        # self.night_idle_bonus = 20.0
        # self.night_idle_miss_penalty = 20.0
        self.night_export_bonus = 10.0
        self.night_export_miss_penalty = 10.0

        # Stronger shaping requested: if SOC <= 0.5 and solar == 0, prefer grid-connected idle.
        # Applied only when grid is available and we're not in a high-risk blackout window.
        self.low_soc_night_idle_reward = 10.0
        self.low_soc_night_idle_miss_penalty = 10.0

        # Cost / wear (tunable)
        self.degradation_cost_coeff = 3.0
        self.action_switch_penalty = 0.3

        # Export shaping (tunable)
        self.export_when_full_bonus = 6.0
        self.export_when_full_miss_penalty = 6.0
        self.export_near_full_margin = 0.05

        # Islanding overhead (tunable)
        # Represents auxiliary loads / conversion overhead when operating islanded.
        # Applied as additional battery draw whenever action is islanding (2 or 3).
        self.islanding_overhead_kw = 2.0

        initial_soc = float(self.df.iloc[0].get("battery_soc", 0.6))
        self.initial_battery_soc = float(np.clip(initial_soc, self.soc_min, self.soc_max))
        self.battery_soc = self.initial_battery_soc

        # -------------------------
        # Action Space
        # -------------------------
        """
        0: Grid-connected (idle)
        1: Grid + charge battery
        2: Islanding (conservative discharge)
        3: Islanding (aggressive discharge)
        4: Safe shutdown
        5: Grid-connected battery discharge with grid export
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
            high=50,
            shape=(len(self.observation_columns),),
            dtype=np.float32
        )

    # -------------------------
    # Helpers
    # -------------------------
    def get_action_mask(self):
        """Return a binary mask over actions (1=allowed, 0=disallowed) for the current timestep.

        This mask is intentionally narrow: it only enforces the user-requested rule that
        battery charging (action 1) is not allowed when solar_actual_kw == 0.
        """

        row = self.df.iloc[self.current_step]
        mask = np.ones(self.action_space.n, dtype=np.int8)

        solar_kw = float(row.get("solar_actual_kw", 0.0))
        if solar_kw <= 0.0:
            mask[1] = 0  # 1: Grid + charge battery

        return mask

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = []
        for col in self.observation_columns:
            if col == "battery_soc":
                obs.append(self.battery_soc)
            else:
                obs.append(row.get(col, 0.0))
        return np.array(obs, dtype=np.float32)

    def _is_action_allowed(self, action, row):
        # Hard safety constraints
        if row["maintenance_active"] == 1:
            # Maintenance implies grid is down; only safe shutdown is allowed.
            return action == 4
        
        if action == 1 and float(row.get("solar_actual_kw", 0.0)) <= 0.0:
            return False

        # Normally: no islanding if grid is up.
        # Exception: if blackout risk is high, preemptive islanding is allowed.
        if row["grid_status"] == 1 and action in [2, 3]:
            return float(row.get("blackout_probability", 0.0)) > 0.6

        # Export action requires grid availability.
        if action == 5:
            return row["grid_status"] == 1

        return True

    def _solar_grid_charge_recommended(self, row, soc_before_action: float, high_risk: bool) -> bool:
        """Return True when grid-connected charging should be preferred due to high solar and SOC headroom."""

        if int(row.get("maintenance_active", 0)) == 1:
            return False

        if int(row.get("grid_status", 1)) != 1:
            return False

        # During high-risk windows, the policy is explicitly encouraged to island instead.
        if high_risk:
            return False

        solar_kw = float(row.get("solar_actual_kw", 0.0))
        # Strictly greater-than threshold ("solar > 2" semantics)
        has_solar_surplus = solar_kw > float(self.solar_charge_threshold_kw)

        # Strict headroom check (battery not full)
        has_soc_headroom = float(soc_before_action) < float(self.soc_max) - float(self.soc_headroom_epsilon)
        return has_solar_surplus and has_soc_headroom

    def _grid_export_recommended(self, row, soc_before_action: float, high_risk: bool) -> bool:
        """Return True when grid-connected export should be preferred (e.g., battery full with solar surplus)."""

        if int(row.get("maintenance_active", 0)) == 1:
            return False

        if int(row.get("grid_status", 1)) != 1:
            return False

        if high_risk:
            return False

        solar_kw = float(row.get("solar_actual_kw", 0.0))
        has_solar_surplus = solar_kw > float(self.solar_charge_threshold_kw)

        # Near-full means the SOC is within a margin of the ceiling.
        near_full_threshold = float(self.soc_max) - float(self.export_near_full_margin)
        is_soc_near_full = float(soc_before_action) >= near_full_threshold
        return has_solar_surplus and is_soc_near_full

    # -------------------------
    # Step
    # -------------------------
    def step(self, action):
        row = self.df.iloc[self.current_step]

        chosen_action = int(action)

        # Hard rule: when solar is zero, charging must be masked out.
        # We *do not* fall back/override here; calling code must respect env.get_action_mask().
        solar_kw = float(row.get("solar_actual_kw", 0.0))
        if int(chosen_action) == 1 and solar_kw <= 0.0:
            raise ValueError(
                "Invalid action: action=1 (Grid + charge battery) while solar_actual_kw == 0. "
                "Use MicrogridEnv.get_action_mask() and mask the policy logits so action 1 cannot be selected."
            )

        soc_before_action = float(self.battery_soc)

        reward = 0.0
        done = False
        info = {}

        # -------- SAFETY OVERRIDE --------
        if not self._is_action_allowed(action, row):
            reward -= 0.0
            action = 4 if int(row["maintenance_active"]) == 1 else 0

        # -------- BLACKOUT-RISK POLICY SHAPING --------
        # If blackout probability is high, the agent should proactively island.
        # Mode selection depends on expected outage duration:
        # - long outage  -> conservative (preserve battery)
        # - short outage -> aggressive (serve more load)
        bp = float(row["blackout_probability"])
        outage_min = float(row.get("expected_outage_duration_min", 0.0))
        high_risk = (bp > 0.6) and (int(row["maintenance_active"]) != 1)

        # Enforce policy constraint: if blackout probability is high, must island.
        # Mode selection depends on expected outage duration.
        if high_risk:
            long_outage_threshold_min = 120.0
            required_island_action = 2 if outage_min >= long_outage_threshold_min else 3
            if int(chosen_action) == int(required_island_action):
                reward += 6.0
            else:
                reward -= 6.0
            action = int(required_island_action)

        # Cache common recommendation flags so reward terms stay consistent.
        charge_recommended = self._solar_grid_charge_recommended(
            row, soc_before_action=soc_before_action, high_risk=high_risk
        )
        export_recommended = self._grid_export_recommended(
            row, soc_before_action=soc_before_action, high_risk=high_risk
        )
        # -------- SOLAR-SURPLUS GRID-CHARGE SHAPING --------
        # When the grid is available and solar is high, prefer charging the battery (if it has headroom).
        # Do not apply while islanding or under high blackout risk.
        if charge_recommended:
            if int(chosen_action) == 1:
                reward += float(self.solar_charge_bonus)
            else:
                reward -= float(self.solar_charge_miss_penalty)

        # -------- BATTERY-FULL GRID-EXPORT SHAPING --------
        # When PV is available but the battery is already full, prefer export rather than idling.
        if export_recommended:
            if int(chosen_action) == 5:
                reward += float(self.export_when_full_bonus)
            elif int(chosen_action) in [0, 1]:
                reward -= float(self.export_when_full_miss_penalty)

        # -------- NIGHT (NO SOLAR) PREFERENCE SHAPING --------
        # If there is effectively no solar (night) and we're in a normal grid-up scenario,
        # prefer idle when SOC is low, and discharge+export when SOC is high.
        if (not high_risk) and int(row.get("maintenance_active", 0)) != 1 and int(row.get("grid_status", 1)) == 1:
            solar_kw = float(row.get("solar_actual_kw", 0.0))
            is_night = solar_kw <= float(self.night_solar_epsilon_kw)
            if is_night:
                # Requested rule: if SOC <= 0.5 and solar == 0, strongly prefer grid-connected idle.
                # Use the agent's chosen action for shaping (not the enforced/safety-overridden action).
                if float(soc_before_action) <= float(self.night_soc_export_threshold):
                    if int(chosen_action) == 0:
                        reward += float(self.low_soc_night_idle_reward)
                    else:
                        reward -= float(self.low_soc_night_idle_miss_penalty)
                else:
                    if int(action) == 5:
                        reward += float(self.night_export_bonus)
                    else:
                        reward -= float(self.night_export_miss_penalty)

        # -------- BATTERY DYNAMICS (DYNAMIC SOC) --------
        # Charging: action 1
        # Discharging: actions 2, 3, 5
        battery_capacity_kwh = float(row.get("battery_capacity_kwh", 20.0))
        battery_max_charge_kw = float(row.get("battery_max_charge_kw", 5.0))
        battery_max_discharge_kw = float(row.get("battery_max_discharge_kw", 5.0))

        target_charge_kw = 0.0
        target_discharge_kw = 0.0

        if action == 1:
            target_charge_kw = battery_max_charge_kw
        elif action == 5:
            target_discharge_kw = battery_max_discharge_kw
        elif action == 2:
            target_discharge_kw = min(2.5, battery_max_discharge_kw)
        elif action == 3:
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

        # Extra SOC drain while islanding (in addition to explicit discharge_kw)
        if action in [2, 3] and float(self.islanding_overhead_kw) > 0.0:
            overhead_draw_kwh = (float(self.islanding_overhead_kw) * self.timestep_hours) / self.discharge_eff
            self.battery_soc = float(np.clip(self.battery_soc - (overhead_draw_kwh / battery_capacity_kwh), self.soc_min, self.soc_max))

        # -------- BATTERY DEGRADATION COST --------
        # Penalize SOC movement (proxy for cycling / wear)
        reward -= abs(float(self.battery_soc) - float(soc_before_action)) * float(self.degradation_cost_coeff)

        # -------- LOADS --------
        load_p1 = row["load_p1_kw"]
        load_p2 = row["load_p2_kw"]
        load_p3 = row["load_p3_kw"]

        total_load_kw = float(load_p1 + load_p2 + load_p3)

        solar = row["solar_actual_kw"]
        battery_soc = self.battery_soc

        # -------- ACTION EFFECTS --------
        is_islanded = action in [2, 3]

        # Action 5 is grid-connected export mode (not islanded)
        is_export = action == 5

        grid_available = (int(row["grid_status"]) == 1) and (int(row["maintenance_active"]) != 1)
        effective_islanded = is_islanded or ((not grid_available) and action != 4)

        # Islanded & load present: choosing an explicit discharge action is preferred.
        # Reward is based on the agent's chosen action (chosen_action), not the enforced action.
        if (not high_risk) and effective_islanded and int(row["maintenance_active"]) != 1 and total_load_kw > 0.0:
            if int(chosen_action) in [2, 3]:
                reward += 6.0
            else:
                reward -= 6.0

        # If the grid is down and we're not explicitly islanding, penalize.
        # (The system will still operate islanded in practice, but we want the policy
        # to choose the appropriate island-mode actions.)
        if (not grid_available) and int(row["maintenance_active"]) != 1 and (action not in [2, 3, 4]):
            reward -= 3.0

        available_power = 0.0
        if effective_islanded:
            available_power = solar + discharge_kw

        # -------- LOAD PRIORITIZATION --------
        served_p1 = served_p2 = served_p3 = 0.0

        if action == 4:
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

        # -------- GRID EXPORT REWARD (ACTION 5) --------
        # When grid is available and action 5 is chosen, export any excess power
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
            reward -= 3.0

        # Positive rewards
        reward += 5.0 * (served_p1 / load_p1)
        reward += 2.0 * (served_p2 / load_p2)
        reward += 0.5 * (served_p3 / load_p3)

        # Penalize unnecessary islanding (when blackout risk is not high)
        if is_islanded and row["grid_status"] == 1 and not high_risk:
            reward -= 6.0

        # Battery protection
        if battery_soc < 0.25:
            reward -= 3.0

        # Maintenance violation (should never happen)
        if row["maintenance_active"] == 1 and is_islanded:
            reward -= 0.0

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
        if (not high_risk) and row["grid_status"] == 1 and low_forecast:
            if int(chosen_action) == 1:
                reward += 6.0
            elif int(chosen_action) in [2, 3, 5]:
                reward -= 6.0

        # Maintenance SOC readiness incentives
        if row["maintenance_active"] == 1:
            if float(battery_soc) < 0.5:
                reward -= 2.0
            elif float(battery_soc) >= 0.7:
                reward += (float(battery_soc) - 0.7)

        # Action flapping penalty (toggle between islanded and grid-connected)
        if self.prev_islanded is not None and bool(self.prev_islanded) != bool(is_islanded):
            reward -= 0.5

        # Action switching penalty (discourage rapid control changes)
        if self.prev_action is not None and int(self.prev_action) != int(action):
            reward -= float(self.action_switch_penalty)

        # -------- STEP FORWARD --------
        self.current_step += 1
        self.prev_islanded = is_islanded
        self.prev_action = action
        if self.current_step >= self.max_steps - 1:
            done = True

        return self._get_obs(), reward, done, info

    # -------------------------
    def reset(self):
        self.current_step = 0
        self.prev_islanded = None
        self.prev_action = None
        self.battery_soc = self.initial_battery_soc
        return self._get_obs()

    def render(self, mode="human"):
        pass
