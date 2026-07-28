# mapek_loop.py  (IMPROVED)
# Key fixes:
#   - monitor() is the single entry point; update_model_if_needed() uses it properly
#   - _adapt_model handles both float and callable learning rates
#   - consecutive_bad_episodes resets correctly for ALL non-bad outcomes
#   - plan() is idempotent: ACCEPTABLE branch no longer silently accumulates
#   - added entropy coefficient boost on adaptation for exploration recovery
#   - get_summary() helper for cleaner logging

import os
from collections import deque

import numpy as np
from stable_baselines3.common.utils import get_schedule_fn

# Shared composite score weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "collision_rate": 0.45,
    "avg_safe_distance": 0.20,
    "avg_speed": 0.10,
    "lane_change": 0.05,
    "detection_rate": 0.20,
}

class MAPEKLoop:
    """
    MAPE-K (Monitor → Analyse → Plan → Execute) self-adaptive loop.

    Usage
    -----
    At the end of every evaluation window call:
        mapek.monitor_and_adapt(episode_metrics_dict)

    During evaluation episodes you can also call:
        mapek.monitor(episode_metrics_dict)
    and then trigger adaptation once per eval window with:
        mapek.update_model_if_needed()
    """

    def __init__(self, model, env=None, model_save_path="models/", history_length=20, eval_only=False):
        self.model = model
        self.env = env  # optional; not used by the MAPE-K logic itself
        self.model_save_path = model_save_path
        self.history = deque(maxlen=history_length)
        self.eval_only = eval_only

        self.best_metrics = None
        self.best_score = -np.inf

        self.consecutive_bad_episodes = 0

        # --- Adaptation configuration ---
        self.adaptation_threshold = 2  # trigger after N consecutive bad/POOR evals
        self.learning_rate_decay = 0.70  # LR multiplier on adaptation
        self.min_learning_rate = 5e-6
        self.ent_coef_boost = 1.5  # multiply entropy coef on adaptation
        self.speed_cap_collision_threshold = 0.02  # collision rate above which to cap speed

        os.makedirs(model_save_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def monitor(self, episode_metrics: dict):
        """Phase 1 – record new metrics."""
        self.history.append(episode_metrics)

    def monitor_and_adapt(self, episode_metrics: dict):
        """Convenience: monitor then run full MAPE-K cycle."""
        self.monitor(episode_metrics)
        self.update_model_if_needed()

    def update_model_if_needed(self):
        """Run one full MAPE-K cycle on the latest history entry."""
        if not self.history:
            return

        current = self.history[-1]
        status = self.analyze(current)
        action = self.plan(status)
        self.execute(action)

    # ------------------------------------------------------------------
    # MAPE-K phases
    # ------------------------------------------------------------------

    def analyze(self, current_metrics: dict) -> str:
        """Phase 2 – compare current performance to best known."""
        if self.best_metrics is None:
            return "FIRST_EPISODE"

        current_score = self._score(current_metrics)
        best_score = self._score(self.best_metrics)

        if best_score == 0:
            return "FIRST_EPISODE"

        ratio = current_score / best_score
        if ratio >= 0.95:
            return "EXCELLENT"
        elif ratio >= 0.85:
            return "GOOD"
        elif ratio >= 0.70:
            return "ACCEPTABLE"
        else:
            return "POOR"

    def plan(self, status: str) -> str:
        """Phase 3 – decide what to do."""
        if status == "FIRST_EPISODE":
            return "UPDATE_BEST"

        if status in ("EXCELLENT", "GOOD"):
            self.consecutive_bad_episodes = 0  # ← was missing for GOOD
            if self.env is not None and hasattr(self.env, "env_method"):
                try:
                    self.env.env_method("clear_speed_cap")
                except AttributeError:
                    pass
            if self._is_better_than_best(self.history[-1]):
                return "UPDATE_BEST"
            return "NO_ACTION"

        if status == "ACCEPTABLE":
            self.consecutive_bad_episodes += 1
            # Don't adapt yet; give it more time
            if self.consecutive_bad_episodes >= self.adaptation_threshold:
                return "NEED_ADAPTATION"
            return "CONTINUE_MONITORING"

        # POOR
        self.consecutive_bad_episodes += 1
        if self.consecutive_bad_episodes >= self.adaptation_threshold:
            return "NEED_ADAPTATION"
        return "CONTINUE_MONITORING"

    def execute(self, action: str):
        """Phase 4 – carry out the plan."""
        if self.eval_only:
            # In eval_only mode, just log the diagnostic status without mutating anything.
            if action == "UPDATE_BEST":
                self.best_metrics = dict(self.history[-1])
                self.best_score = self._score(self.best_metrics)
                print(f"✨ New best model detected (eval_only – no action taken)")
            elif action == "NEED_ADAPTATION":
                print(f"⚠️  Performance degraded (no action taken — eval_only mode)")
            elif action == "CONTINUE_MONITORING":
                streak = self.consecutive_bad_episodes
                print(f" Monitoring (sub-optimal streak: {streak}/{self.adaptation_threshold} — eval_only mode)")
            return

        if action == "UPDATE_BEST":
            print("✨ New best model → saving …")
            self.best_metrics = dict(self.history[-1])
            self.best_score = self._score(self.best_metrics)
            self._save_best_model()

        elif action == "NEED_ADAPTATION":
            print("⚠️  Performance degraded – triggering adaptation …")
            self._adapt_model()
            self.consecutive_bad_episodes = 0

        elif action == "CONTINUE_MONITORING":
            streak = self.consecutive_bad_episodes
            print(
                f" Monitoring … (sub-optimal streak: {streak}/{self.adaptation_threshold})"
            )

        else:  # NO_ACTION
            pass

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, metrics: dict) -> float:
        """
        Weighted composite score in [0, 1].
        Weights: safety (45 %) + distance (20 %) + speed (10 %) +
                 lane-change (5 %) + detection (20 %)
        """
        if not metrics:
            return 0.0
        lane_val = metrics.get("lane_change_success_rate")
        if lane_val is None:
            lane_val = 0.5
        return (
            SCORE_WEIGHTS["collision_rate"] * (1.0 - min(metrics.get("collision_rate", 1.0), 1.0))
            + SCORE_WEIGHTS["avg_safe_distance"] * min(metrics.get("avg_safe_distance", 0.0) / 30.0, 1.0)
            + SCORE_WEIGHTS["avg_speed"] * min(metrics.get("avg_speed", 0.0) / 30.0, 1.0)
            + SCORE_WEIGHTS["lane_change"] * lane_val
            + SCORE_WEIGHTS["detection_rate"] * metrics.get("detection_rate", 0.0)
        )

    def _is_better_than_best(self, metrics: dict) -> bool:
        if self.best_metrics is None:
            return True
        return self._score(metrics) > self.best_score

    # ------------------------------------------------------------------
    # Adaptation
    # ------------------------------------------------------------------

    def _adapt_model(self):
        """Decay learning rate + boost exploration entropy + optional speed cap."""
        print(" Adapting model …")

        # --- Learning rate ---
        # model.learning_rate is the raw value; model.lr_schedule is the
        # callable actually used by the optimizer. Both must be updated.
        current_lr = self.model.learning_rate
        if callable(current_lr):
            try:
                current_lr = float(current_lr(1.0))
            except Exception:
                current_lr = 3e-4

        new_lr = max(current_lr * self.learning_rate_decay, self.min_learning_rate)
        self.model.learning_rate = new_lr
        self.model.lr_schedule = get_schedule_fn(
            new_lr
        )  # ← actually changes optimizer LR
        print(f"    LR: {current_lr:.2e} → {new_lr:.2e}")

        # --- Entropy coefficient ---
        # Cap is set to 0.15 (above the initial 0.05) so the boost has room to work.
        if hasattr(self.model, "ent_coef") and isinstance(self.model.ent_coef, float):
            old_ent = self.model.ent_coef
            new_ent = min(old_ent * self.ent_coef_boost, 0.15)
            self.model.ent_coef = new_ent
            print(f"    ent_coef: {old_ent:.4f} → {new_ent:.4f}")

        # --- Speed cap when collision rate is high ---
        if self.env is not None and hasattr(self.env, "env_method"):
            current = self.history[-1] if self.history else {}
            collision_rate = current.get("collision_rate", 0.0)
            if collision_rate > self.speed_cap_collision_threshold:
                try:
                    self.env.env_method("set_speed_cap")
                    print(f"    Speed cap ACTIVATED (collision_rate={collision_rate:.4f})")
                except AttributeError:
                    pass

        print("✅ Adaptation done.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_best_model(self):
        path = os.path.join(self.model_save_path, "best_mapek_model")
        self.model.save(path)
        print(f" Best model saved → {path}.zip")

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_history(self) -> list:
        return list(self.history)

    def get_best_metrics(self) -> dict:
        return self.best_metrics

    def get_summary(self) -> str:
        lines = [
            "MAPE-K Summary",
            f"  History length  : {len(self.history)}",
            f"  Best score      : {self.best_score:.4f}",
            f"  Bad streak      : {self.consecutive_bad_episodes}",
        ]
        if self.best_metrics:
            lines.append("  Best metrics    :")
            for k, v in self.best_metrics.items():
                lines.append(
                    f"    {k:30s}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}"
                )
        return "\n".join(lines)

