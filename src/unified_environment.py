# unified_environment.py
# Key fixes (v3):
#   - Markov attack model: attacks persist across steps (p_continue=0.85)
#     so the LSTM has a learnable temporal pattern instead of i.i.d. noise
#   - Fixed timing: spoofing for step t is pre-sampled at end of step t-1
#     so obs_t encodes the attack state the agent will actually face at t
#   - Decoupled GPS from mitigation: GPS indicator is a raw sensor reading
#     (independent of _mitigated) so the signal is causal and actionable
#   - Noise simplified: only depends on _spoofing, not _mitigated
#   - All other fixes from v2 preserved

import random

import gymnasium as gym
import numpy as np


class UnifiedEnvironment(gym.Wrapper):
    """
    Wraps highway-v0 and adds a GPS-spoofing attack layer.

    Observation  : flattened highway obs (25-d) + 1 GPS integrity signal + 1 GPS EMA = 27-d
    Action space : MultiDiscrete([5, 2])
                     dim-0 → driving action  (0-4, same as highway-v0)
                     dim-1 → mitigate_spoof  (0 = no, 1 = yes)
    """

    # highway-env discrete action meanings
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

    SAFE_DISTANCE_THRESHOLD = 20.0
    SPEED_PENALTY_FACTOR = 0.05

    def __init__(
        self,
        env,
        gps_spoofing_prob: float = 0.30,
        safe_distance_threshold: float = 20.0,
        speed_penalty_factor: float = 0.05,
    ):
        # gps_spoofing_prob is kept for backward-compatible call sites but is
        # no longer used — attack timing is governed by the Markov chain
        # (p_attack_start / p_attack_continue) defined below.
        super().__init__(env)

        # --- observation space (flattened + 1 GPS indicator) ---
        original_shape = self.env.observation_space.shape  # e.g. (5, 5)
        self._obs_flat = int(np.prod(original_shape))  # 25
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_flat + 2,),
            dtype=np.float32,
        )

        # --- action space ---
        self.action_space = gym.spaces.MultiDiscrete([self.env.action_space.n, 2])

        # --- attack parameters (Markov chain) ---
        # Steady-state attack rate ≈ p_start / (p_start + p_stop)
        #                         ≈ 0.05 / (0.05 + 0.15) ≈ 25% of steps
        # Mean attack burst length ≈ 1 / (1 - p_continue) ≈ 7 steps
        # This gives the LSTM a learnable temporal pattern.
        self.p_attack_start = 0.05  # prob. a NEW attack begins at a clean step
        self.p_attack_continue = 0.85  # prob. an ONGOING attack persists

        self.safe_distance_threshold = safe_distance_threshold
        self.speed_penalty_factor = speed_penalty_factor
        self._speed_cap_active = False

        self._gps_ema = 0.75
        self._gps_ema_alpha = 0.3

        # --- internal state ---
        self._spoofing = False  # current step's spoofing state (pre-sampled)
        self._attack_ongoing = False  # Markov chain state
        self._mitigated = False
        self._episode = 0
        self._reset_metrics()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._mitigated = False
        self._attack_ongoing = False
        self._episode += 1
        self._reset_metrics()

        # Pre-sample the spoofing state for the FIRST step so obs_0
        # already carries an actionable GPS signal.
        self._spoofing = random.random() < self.p_attack_start
        self._attack_ongoing = self._spoofing
        self._gps_ema = 0.75

        return self._build_obs(obs), info

    def step(self, action):
        driving_action, mitigate_action = int(action[0]), int(action[1])
        self._mitigated = mitigate_action == 1

        # ---- Security reward ----
        # _spoofing was pre-sampled at the end of the previous step (or at reset),
        # so the GPS indicator in obs already encoded this state and the agent
        # had a fair chance to decide whether to mitigate.
        if self._mitigated and self._spoofing:
            security_reward = +2.0  # True Positive
        elif self._mitigated and not self._spoofing:
            security_reward = -2.0  # False Positive
            self._metrics["false_positives"] += 1
        elif not self._mitigated and self._spoofing:
            security_reward = -2.0  # False Negative (missed attack)
        else:
            security_reward = +0.5  # True Negative

        # ---- Track lane before step ----
        prev_lane = self._current_lane()

        # ---- Execute driving action ----
        true_obs, driving_reward, terminated, truncated, info = self.env.step(
            driving_action
        )

        # ---- Augment info ----
        env_uw = self.env.unwrapped
        vehicle = getattr(env_uw, "vehicle", None)

        info["speed"] = vehicle.speed if vehicle and hasattr(vehicle, "speed") else 0.0
        info["distance_to_nearest_car"] = self._safe_distance()
        info["spoofing_active"] = self._spoofing
        info["mitigation_active"] = self._mitigated

        crashed = info.get("crashed", False)

        # ---- Update metrics ----
        self._metrics["total_steps"] += 1
        self._metrics["total_speed"] += info["speed"]
        self._metrics["total_safe_distance"] += info["distance_to_nearest_car"]

        if crashed:
            self._metrics["collisions"] += 1

        if self._spoofing:
            self._metrics["spoofed_steps"] += 1
        if self._spoofing and self._mitigated:
            self._metrics["detected_steps"] += 1

        # Lane change tracking (correct indices: 0=LEFT, 2=RIGHT)
        new_lane = self._current_lane()
        lane_change_succeeded = False
        if driving_action in (self.LANE_LEFT, self.LANE_RIGHT):
            self._metrics["lane_attempts"] += 1
            if (
                new_lane is not None
                and prev_lane is not None
                and new_lane != prev_lane
                and not crashed
            ):
                self._metrics["successful_lane_changes"] += 1
                lane_change_succeeded = True

        # Small reward for successful lane changes — makes occasional safe
        # lane changes worth trying instead of never attempting them.
        lane_change_bonus = 0.15 if lane_change_succeeded else 0.0

        # ---- Combined reward ----
        # Driving remains the primary objective; security is a secondary term.
        # When under attack the security signal carries more weight so the
        # agent has an incentive to mitigate, but driving is never below 60 %.
        drive_w = 0.70 if not self._spoofing else 0.60
        security_w = 1.0 - drive_w

        # Small penalty for choosing IDLE — discourages the degenerate
        # strategy of sitting still and ignoring the driving task entirely.
        idle_penalty = -0.3 if driving_action == self.IDLE else 0.0

        distance_penalty = 0.0
        if info["distance_to_nearest_car"] < self.safe_distance_threshold:
            penalty_factor = self.speed_penalty_factor * (2.0 if self._speed_cap_active else 1.0)
            distance_penalty = -penalty_factor * info["speed"]

        total_reward = (
            drive_w * driving_reward + security_w * security_reward + idle_penalty + distance_penalty + lane_change_bonus
        )

        # ---- Advance Markov chain: sample NEXT step's spoofing state ----
        # The resulting _spoofing will be encoded in next_obs so the agent
        # sees the GPS integrity signal BEFORE deciding whether to mitigate.
        if self._attack_ongoing:
            self._spoofing = random.random() < self.p_attack_continue
        else:
            self._spoofing = random.random() < self.p_attack_start
        self._attack_ongoing = self._spoofing

        next_obs = self._build_obs(true_obs)
        return next_obs, total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, true_obs: np.ndarray) -> np.ndarray:
        flat = true_obs.flatten().astype(np.float32)

        # Observation noise: based purely on whether the NEXT step is under
        # attack (_spoofing was just updated by the Markov chain).
        # Mitigation is NOT factored in here — the agent must learn to mitigate
        # based on the GPS signal, not the noise alone.
        if self._spoofing:
            flat = flat + np.random.normal(0, 0.12, flat.shape).astype(np.float32)
        # else: clean — no noise

        # ---- Raw GPS integrity indicator ----
        # Distributions deliberately overlap so a single reading is ambiguous.
        # The LSTM must accumulate multiple steps to decide reliably — this
        # forces it to actually use its temporal memory.
        #
        #   Clean:   N(0.75, 0.15)  →  range roughly [0.35 – 1.0]
        #   Spoofed: N(0.40, 0.15)  →  range roughly [0.0  – 0.75]
        #   Overlap zone ≈ [0.35 – 0.75]: ambiguous on any single step
        if self._spoofing:
            gps_val = np.random.normal(0.40, 0.15)
        else:
            gps_val = np.random.normal(0.75, 0.15)

        gps_val = float(np.clip(gps_val, 0.0, 1.0))
        self._gps_ema = self._gps_ema_alpha * gps_val + (1 - self._gps_ema_alpha) * self._gps_ema
        return np.append(flat, [gps_val, self._gps_ema]).astype(np.float32)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _reset_metrics(self):
        self._metrics = {
            "total_steps": 0,
            "collisions": 0,
            "total_speed": 0.0,
            "total_safe_distance": 0.0,
            "spoofed_steps": 0,
            "detected_steps": 0,
            "lane_attempts": 0,
            "successful_lane_changes": 0,
            "false_positives": 0,
        }

    def get_metrics(self) -> dict:
        m = self._metrics
        ts = max(m["total_steps"], 1)

        # collision_rate = collisions per episode (not per step)
        # For a single-episode call this equals 0 or 1 (or more if rare).
        # Normalise by episode steps so values stay in [0, 1].
        collision_rate = m["collisions"] / ts

        avg_safe_distance = m["total_safe_distance"] / ts
        avg_speed = m["total_speed"] / ts

        lc_success = (
            m["successful_lane_changes"] / m["lane_attempts"]
            if m["lane_attempts"] > 0
            else None
        )

        detection_rate = (
            m["detected_steps"] / m["spoofed_steps"] if m["spoofed_steps"] > 0 else 1.0
        )

        non_spoofed = ts - m["spoofed_steps"]
        fp_rate = m["false_positives"] / non_spoofed if non_spoofed > 0 else 0.0

        return {
            "collision_rate": collision_rate,
            "avg_safe_distance": avg_safe_distance,
            "avg_speed": avg_speed,
            "lane_change_success_rate": lc_success,
            "lane_attempts": m["lane_attempts"],
            "detection_rate": detection_rate,
            "false_positive_rate": fp_rate,
            # raw counts (useful for debugging)
            "total_steps": ts,
            "spoofed_steps": m["spoofed_steps"],
            "collisions_raw": m["collisions"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_lane(self):
        env_uw = self.env.unwrapped
        vehicle = getattr(env_uw, "vehicle", None)
        if vehicle and hasattr(vehicle, "lane_index"):
            li = vehicle.lane_index
            if li and len(li) > 2:
                return li[2]
        return None

    def _safe_distance(self) -> float:
        env_uw = self.env.unwrapped
        if not hasattr(env_uw, "vehicle") or not hasattr(env_uw, "road"):
            return 0.0  # 0 = unknown, not 30 (avoids flat line)

        ego = env_uw.vehicle
        if not hasattr(ego, "position"):
            return 0.0

        ego_pos = np.array(ego.position)
        min_dist = float("inf")

        for other in env_uw.road.vehicles:
            if other is not ego and hasattr(other, "position"):
                d = float(np.linalg.norm(ego_pos - np.array(other.position)))
                if d < min_dist:
                    min_dist = d

        return min_dist if min_dist != float("inf") else 0.0

    def set_speed_cap(self):
        """Activate 2x speed penalty when driving too close (called by MAPE-K)."""
        print(f"    [UnifiedEnvironment] set_speed_cap called on env id={id(self)}")
        self._speed_cap_active = True

    def clear_speed_cap(self):
        """Deactivate speed penalty multiplier (performance recovered)."""
        self._speed_cap_active = False

    def render(self):
        return self.env.render()

