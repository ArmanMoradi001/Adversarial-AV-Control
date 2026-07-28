# train_unified_agent.py  (v3 — curriculum learning + explicit GPU support)
# Changes from v2:
#   - Curriculum: 4 phases (easy → medium → target → stabilization)
#     so the LSTM builds detection skill progressively
#   - make_env() is now a factory-factory accepting p_continue per phase
#   - Explicit GPU detection with diagnostics; falls back to CPU cleanly
#   - ent_coef raised to 0.05 (more exploration of the mitigate action early on)
#   - Training plots shade phase boundaries for easier analysis
#   - train_log records phase number and p_continue per iteration

import json
import os

import gymnasium as gym
import highway_env  # registers highway-v0 automatically on import
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tqdm import tqdm

from mapek_loop import MAPEKLoop
from unified_environment import UnifiedEnvironment

# ======================================================================
# Curriculum schedule
#
# Each phase trains for `timesteps` steps with attacks that persist for
# roughly  1 / (1 − p_continue)  steps on average:
#
#   Phase 1  p_continue=0.98  →  ~50 steps / burst   (trivially detectable)
#   Phase 2  p_continue=0.92  →  ~12 steps / burst   (medium)
#   Phase 3  p_continue=0.85  →   ~7 steps / burst   (target difficulty)
#   Phase 4  p_continue=0.85  →   ~7 steps / burst   (stabilization)
#
# The agent first learns "GPS low → mitigate" on long attacks, then
# gradually adapts to shorter, harder-to-catch bursts, then stabilises.
# ======================================================================

CURRICULUM = [
    dict(
        timesteps=500_000, p_continue=0.98, label="Phase 1 — Easy   (~50-step bursts)"
    ),
    dict(
        timesteps=700_000, p_continue=0.92, label="Phase 2 — Medium (~12-step bursts)"
    ),
    dict(
        timesteps=1_300_000, p_continue=0.85, label="Phase 3 — Target  (~7-step bursts)"
    ),
    dict(
        timesteps=500_000, p_continue=0.85,
        label="Phase 4 — Stabilization (same difficulty, safety-focused)"
    ),
]


# ======================================================================
# Feature extractor (unchanged)
# ======================================================================


class FlattenExtractor(BaseFeaturesExtractor):
    """MLP feature extractor for the 27-d flat observation."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)
        input_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(input_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# ======================================================================
# MAPE-K callback
# ======================================================================


class MAPEKCallback(BaseCallback):
    def __init__(self, mapek: MAPEKLoop, eval_env, verbose: int = 0):
        super().__init__(verbose)
        self.mapek = mapek
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def feed_metrics(self, metrics: dict):
        self.mapek.monitor_and_adapt(metrics)


# ======================================================================
# Env factory
# ======================================================================


def make_env(p_continue: float = 0.85):
    """
    Returns an env factory (callable → env) for use with make_vec_env.
    p_continue sets the Markov attack persistence probability.
    """

    def _init():
        env = gym.make("highway-v0")
        w = UnifiedEnvironment(env)
        w.p_attack_continue = p_continue
        return w

    return _init


# ======================================================================
# Eval helper
# ======================================================================


def evaluate_and_get_metrics(
    model, n_episodes: int = 5, p_continue: float = 0.85
) -> dict:
    """
    Run n_episodes on a plain (non-vectorised) env and return mean metrics.
    p_continue should match the current curriculum phase.
    """
    all_metrics = []
    raw_env = make_env(p_continue)()  # call the factory to get an env instance

    for _ in range(n_episodes):
        obs, _ = raw_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False

        while not done:
            action, lstm_states = model.predict(
                obs[np.newaxis],
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _, terminated, truncated, _ = raw_env.step(action[0])
            done = terminated or truncated
            episode_starts = np.array([done])

        all_metrics.append(raw_env.get_metrics())

    raw_env.close()

    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}


# ======================================================================
# Plotting helper
# ======================================================================


def save_training_plots(
    train_log: list, curriculum: list, save_dir: str = "training_results"
):
    """Save per-metric training curves with phase boundaries shaded."""
    os.makedirs(save_dir, exist_ok=True)

    steps = [d["timesteps"] for d in train_log]

    # Compute phase x-boundaries for shading
    phase_bounds = []
    start = 0
    for phase in curriculum:
        end = start + int(phase["timesteps"])
        phase_bounds.append((start, end, phase["label"]))
        start = end

    phase_colors = ["#cce5ff", "#fff3cd", "#d4edda", "#e8daef"]  # blue / yellow / green / lavender

    plots = [
        ("detection_rate", "Detection Rate", "cyan", "Rate"),
        ("false_positive_rate", "False Positive Rate", "orange", "Rate"),
        ("collision_rate", "Collision Rate", "red", "Rate"),
        ("avg_speed", "Avg Speed (m/s)", "royalblue", "m/s"),
        ("avg_safe_distance", "Avg Safe Distance (m)", "green", "Metres"),
        ("learning_rate", "Learning Rate", "purple", "LR"),
    ]

    for key, title, color, ylabel in plots:
        fig, ax = plt.subplots(figsize=(12, 4))

        # Shade curriculum phases
        for i, (x0, x1, lbl) in enumerate(phase_bounds):
            ax.axvspan(
                x0, x1, alpha=0.25, color=phase_colors[i % len(phase_colors)], label=lbl
            )
            ax.axvline(x0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

        values = [d[key] for d in train_log]
        ax.plot(steps, values, color=color, linewidth=2, marker="o", markersize=3)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{save_dir}/{key}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")


# ======================================================================
# Main training script
# ======================================================================

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("unified_ppo_tensorboard", exist_ok=True)
    os.makedirs("training_results", exist_ok=True)

    # ---- GPU / CPU detection ----
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
        device_info = f"CUDA — {gpu_name} ({gpu_vram} GB VRAM)"
    else:
        device = "cpu"
        device_info = (
            "CPU  (no CUDA GPU found)\n"
            "  Tip: install GPU PyTorch with:\n"
            "       pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    # ---- Config ----
    n_cpu = 4
    eval_freq = 50_000
    n_eval_episodes_sb3 = 2  # used by EvalCallback (fast SB3 best-model tracking)
    n_eval_episodes_mapek = (
        10  # used by MAPE-K (needs more episodes for reliable signal)
    )
    total_timesteps = sum(int(p["timesteps"]) for p in CURRICULUM)

    print("=" * 70)
    print("  Training Secure Autonomous Agent — Curriculum + MAPE-K")
    print("=" * 70)
    print(f"  Device          : {device_info}")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Eval every      : {eval_freq:,} steps")
    print(f"  Parallel envs   : {n_cpu}")
    print("-" * 70)
    for p in CURRICULUM:
        print(f"    {p['label']}  —  {p['timesteps']:>9,} steps")
    print("=" * 70)

    # ---- Build initial environments (Phase 1 difficulty) ----
    first_p = float(CURRICULUM[0]["p_continue"])
    try:
        train_env = make_vec_env(
            make_env(first_p), n_envs=n_cpu, vec_env_cls=SubprocVecEnv
        )
    except Exception:
        print("  SubprocVecEnv failed — falling back to DummyVecEnv")
        train_env = make_vec_env(
            make_env(first_p), n_envs=n_cpu, vec_env_cls=DummyVecEnv
        )

    eval_env = DummyVecEnv([make_env(first_p)])

    # ---- Policy ----
    policy_kwargs = dict(
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        lstm_hidden_size=128,
        n_lstm_layers=1,
        share_features_extractor=True,
    )

    # ---- Model ----
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=512,  # per env → total batch = 512 × n_cpu = 2048
        batch_size=128,
        n_epochs=10,
        gamma=0.97,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # higher than before: encourages exploring mitigate=1 early
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./unified_ppo_tensorboard/",
        device=device,  # explicit CUDA or CPU
    )

    # ---- MAPE-K ----
    # Pass train_env so speed-cap adaptation can reach the underlying envs.
    print("\n  Initialising MAPE-K loop …")
    mapek = MAPEKLoop(model, train_env, model_save_path="models/")

    # ---- Checkpoint callback (shared across phases) ----
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // n_cpu, 1),
        save_path="./models/",
        name_prefix="recurrent_ppo",
    )

    # ======================================================================
    # Curriculum training loop
    # ======================================================================
    train_log = []
    global_steps = 0

    for phase_idx, phase in enumerate(CURRICULUM):
        p_continue = float(phase["p_continue"])
        phase_steps = int(phase["timesteps"])
        n_iters = phase_steps // eval_freq

        print(f"\n{'=' * 70}")
        print(f"  {phase['label']}")
        print(f"  Iterations : {n_iters}  |  Steps : {phase_steps:,}")
        print(f"{'=' * 70}")

        # Swap environments to current phase difficulty (skip for first phase)
        if phase_idx > 0:
            train_env.close()
            eval_env.close()
            try:
                train_env = make_vec_env(
                    make_env(p_continue), n_envs=n_cpu, vec_env_cls=SubprocVecEnv
                )
            except Exception:
                train_env = make_vec_env(
                    make_env(p_continue), n_envs=n_cpu, vec_env_cls=DummyVecEnv
                )
            eval_env = DummyVecEnv([make_env(p_continue)])
            model.set_env(train_env)  # attach new env to existing model weights
            mapek.env = train_env  # keep MAPEK speed-cap pointed at current envs
            print(f"  ✅ Environments swapped to p_continue={p_continue}")

        # Fresh EvalCallback for this phase (points at the current eval_env)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/best_model/",
            log_path="./logs/",
            eval_freq=eval_freq // n_cpu,
            deterministic=True,
            render=False,
            n_eval_episodes=n_eval_episodes_sb3,
        )

        pbar = tqdm(
            range(n_iters),
            desc=f"Phase {phase_idx + 1}/{len(CURRICULUM)}",
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for _ in pbar:
            global_steps += eval_freq

            model.learn(
                total_timesteps=eval_freq,
                callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=False,
                progress_bar=False,
            )

            metrics = evaluate_and_get_metrics(model, n_eval_episodes_mapek, p_continue)
            mapek.monitor_and_adapt(metrics)

            current_lr = model.learning_rate
            if callable(current_lr):
                try:
                    current_lr = float(current_lr(1.0))  # type: ignore[arg-type]
                except Exception:
                    current_lr = 3e-4

            train_log.append(
                {
                    "timesteps": global_steps,
                    "phase": phase_idx + 1,
                    "p_continue": p_continue,
                    "collision_rate": metrics.get("collision_rate", 0),
                    "detection_rate": metrics.get("detection_rate", 0),
                    "false_positive_rate": metrics.get("false_positive_rate", 0),
                    "avg_speed": metrics.get("avg_speed", 0),
                    "avg_safe_distance": metrics.get("avg_safe_distance", 0),
                    "learning_rate": current_lr,
                }
            )

            pbar.set_postfix(
                {
                    "detect": f"{metrics.get('detection_rate', 0):.3f}",
                    "fp": f"{metrics.get('false_positive_rate', 0):.3f}",
                    "crash": f"{metrics.get('collision_rate', 0):.3f}",
                    "speed": f"{metrics.get('avg_speed', 0):.1f}",
                }
            )

    # ---- Save final model ----
    final_path = "models/unified_ppo_model_final"
    model.save(final_path)
    print(f"\n✅ Training complete. Model saved → {final_path}.zip")
    print(mapek.get_summary())

    # ---- Save training log ----
    log_path = "training_results/training_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=4)
    print(f"  Log saved → {log_path}")

    # ---- Save plots ----
    save_training_plots(train_log, CURRICULUM)

    train_env.close()
    eval_env.close()

