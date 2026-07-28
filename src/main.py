# main.py  (IMPROVED)
# Key fixes:
#   - mapek.monitor() used correctly (not raw history.append)
#   - plots use only English labels (no Persian font issues)
#   - detection_rate / fp_rate averaged across episodes to avoid binary flatline
#   - performance score plotted with rolling average for readability
#   - model load path falls back gracefully
#   - results JSON schema unchanged (compatible with old reader code)

import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"  # safe cross-platform font

import gymnasium as gym
import highway_env  # registers highway-v0 automatically on import
from sb3_contrib import RecurrentPPO

from mapek_loop import MAPEKLoop, SCORE_WEIGHTS
from unified_environment import UnifiedEnvironment

# ======================================================================
# Model loading
# ======================================================================


# Use best_mapek_model instead of unified_ppo_model_final: training logs showed the final
# checkpoint had the highest collision rate of the whole run, so the best-tracked checkpoint
# should be used for evaluation and reporting.
def load_model(model_path: str = "models/best_mapek_model") -> RecurrentPPO:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device.upper()}")
    try:
        model = RecurrentPPO.load(model_path, device=device)
        print(f"✅ Loaded trained model from {model_path}.zip")
        return model
    except Exception as e:
        print(f"⚠️  Could not load trained model: {e}")
        print(" Creating untrained dummy model for testing …")
        env = UnifiedEnvironment(gym.make("highway-v0"))
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, device=device)
        env.close()
        print("✅ Dummy model created (results will be random-policy quality).")
        return model


# ======================================================================
# Plotting & saving
# ======================================================================


def _rolling(values: list, window: int = 30) -> np.ndarray:
    """Simple centred rolling average (pads edges with edge values)."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(arr)]


def plot_performance(history: list, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)

    if not history:
        print("No history to plot.")
        return

    episodes = list(range(1, len(history) + 1))

    collision_rates = [m["collision_rate"] for m in history]
    avg_safe_distances = [m["avg_safe_distance"] for m in history]
    avg_speeds = [m["avg_speed"] for m in history]
    lane_success = [m["lane_change_success_rate"] for m in history]
    lane_success_plot = [x if x is not None else np.nan for x in lane_success]
    detection_rates = [m["detection_rate"] for m in history]
    false_positive_rates = [m.get("false_positive_rate", 0.0) for m in history]

    episodes_with_lane_attempts = sum(
        1 for m in history if m.get("lane_attempts", 0) > 0
    )

    performance_scores = []
    for m in history:
        lane_val = m["lane_change_success_rate"]
        if lane_val is None:
            lane_val = 0.5
        s = (
            SCORE_WEIGHTS["collision_rate"] * (1.0 - min(m["collision_rate"], 1.0))
            + SCORE_WEIGHTS["avg_safe_distance"] * min(m["avg_safe_distance"] / 30.0, 1.0)
            + SCORE_WEIGHTS["avg_speed"] * min(m["avg_speed"] / 30.0, 1.0)
            + SCORE_WEIGHTS["lane_change"] * lane_val
            + SCORE_WEIGHTS["detection_rate"] * m["detection_rate"]
        )
        performance_scores.append(s)

    # ---- Save JSON ----
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_episodes": len(history),
        "episodes": episodes,
        "collision_rates": collision_rates,
        "avg_safe_distance": avg_safe_distances,
        "avg_speed": avg_speeds,
        "lane_change_success_rate": lane_success,
        "detection_rate": detection_rates,
        "false_positive_rate": false_positive_rates,
        "performance_scores": performance_scores,
        "statistics": {
            "mean_collision_rate": float(np.mean(collision_rates)),
            "std_collision_rate": float(np.std(collision_rates)),
            "mean_detection_rate": float(np.mean(detection_rates)),
            "std_detection_rate": float(np.std(detection_rates)),
            "mean_speed": float(np.mean(avg_speeds)),
            "mean_safe_distance": float(np.mean(avg_safe_distances)),
            "mean_lane_change_success": float(
                np.mean([x for x in lane_success if x is not None])
            ) if any(x is not None for x in lane_success) else None,
            "pct_episodes_with_lane_attempts": episodes_with_lane_attempts / len(history),
            "mean_false_positive_rate": float(np.mean(false_positive_rates)),
            "mean_performance_score": float(np.mean(performance_scores)),
        },
    }

    json_path = f"{save_dir}/experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f" Results saved → {json_path}")

    # ---- Individual plots ----
    def _save_plot(
        y,
        label,
        color,
        filename,
        ylabel="Rate",
        ylim=None,
        y2=None,
        y2_label=None,
        y2_color=None,
    ):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(episodes, y, color=color, alpha=0.25, linewidth=1, label="raw")
        if len(y) >= 3:
            ax.plot(
                episodes,
                _rolling(y),
                color=color,
                linewidth=2,
                label="smoothed (30-ep)",
            )
        if y2 is not None:
            ax.plot(episodes, y2, color=y2_color, alpha=0.25, linewidth=1)
            if len(y2) >= 3:
                ax.plot(
                    episodes, _rolling(y2), color=y2_color, linewidth=2, label=y2_label
                )
        ax.set_title(label, fontweight="bold", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{save_dir}/{filename}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f" Saved → {path}")

    _save_plot(
        collision_rates, "Collision Rate", "red", "01_collision_rate", ylim=(0, None)
    )
    _save_plot(
        avg_safe_distances,
        "Avg Safe Distance (m)",
        "green",
        "02_safe_distance",
        ylabel="Metres",
        ylim=(0, None),
    )
    _save_plot(
        avg_speeds,
        "Avg Speed (m/s)",
        "royalblue",
        "03_avg_speed",
        ylabel="m/s",
        ylim=(0, None),
    )
    _save_plot(
        lane_success_plot,
        "Lane Change Success Rate",
        "purple",
        "04_lane_change_success",
        ylim=(0, 1.05),
    )
    _save_plot(
        detection_rates,
        "Attack Detection vs False Positives",
        "cyan",
        "05_detection_vs_fp",
        ylim=(0, 1.05),
        y2=false_positive_rates,
        y2_label="false positives",
        y2_color="orange",
    )
    _save_plot(
        performance_scores,
        "Composite Performance Score",
        "black",
        "06_performance_score",
        ylim=(0, 1.05),
    )

    # ---- Combined overview (still saved too) ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Agent Performance under MAPE-K Adaptation", fontsize=15, fontweight="bold"
    )

    def _plot(ax, y, label, color, ylabel="Rate", ylim=None, smooth=True):
        ax.plot(episodes, y, color=color, alpha=0.25, linewidth=1, label="raw")
        if smooth and len(y) >= 3:
            ax.plot(episodes, _rolling(y), color=color, linewidth=2, label="smoothed")
            ax.legend(fontsize=8)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], collision_rates, "Collision Rate", "red", ylim=(0, None))
    _plot(
        axes[0, 1],
        avg_safe_distances,
        "Avg Safe Distance (m)",
        "green",
        ylabel="Metres",
        ylim=(0, None),
    )
    _plot(
        axes[0, 2],
        avg_speeds,
        "Avg Speed (m/s)",
        "royalblue",
        ylabel="m/s",
        ylim=(0, None),
    )
    _plot(
        axes[1, 0], lane_success_plot, "Lane Change Success Rate", "purple", ylim=(0, 1.05)
    )

    ax = axes[1, 1]
    ax.plot(episodes, detection_rates, color="cyan", alpha=0.25, linewidth=1)
    ax.plot(episodes, false_positive_rates, color="orange", alpha=0.25, linewidth=1)
    if len(episodes) >= 3:
        ax.plot(
            episodes,
            _rolling(detection_rates),
            color="cyan",
            linewidth=2,
            label="detection",
        )
        ax.plot(
            episodes,
            _rolling(false_positive_rates),
            color="orange",
            linewidth=2,
            label="false positives",
        )
    ax.set_title("Attack Detection vs False Positives", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _plot(
        axes[1, 2],
        performance_scores,
        "Composite Performance Score",
        "black",
        ylim=(0, 1.05),
    )

    plt.tight_layout()
    plot_path = f"{save_dir}/00_all_metrics_overview.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Saved → {plot_path}")

    # ---- Console summary ----
    s = results["statistics"]
    print("\n" + "=" * 70)
    print("  STATISTICS SUMMARY")
    print("=" * 70)
    print(
        f"   Collision rate (mean ± std)  : {s['mean_collision_rate']:.4f} ± {s['std_collision_rate']:.4f}"
    )
    print(
        f"   Detection rate (mean ± std)  : {s['mean_detection_rate']:.4f} ± {s['std_detection_rate']:.4f}"
    )
    print(f"   False positive rate (mean)   : {s['mean_false_positive_rate']:.4f}")
    print(f"   Avg speed (mean)             : {s['mean_speed']:.2f} m/s")
    print(f"   Avg safe distance (mean)     : {s['mean_safe_distance']:.2f} m")
    if s['mean_lane_change_success'] is not None:
        print(
            f"   Lane change success (mean)   : "
            f"{s['mean_lane_change_success']:.3f}"
        )
    else:
        print("   Lane change success (mean)   : N/A (no lane attempts)")
    print(f"   Pct episodes with lane att.  : {s['pct_episodes_with_lane_attempts']:.1%}")
    print(f"   Composite score (mean)       : {s['mean_performance_score']:.4f}")
    print("=" * 70)


# ======================================================================
# Main evaluation loop
# ======================================================================


def main():
    print("=" * 70)
    print("  Secure Autonomous Driving – GPS Spoofing Detection")
    print("    Self-Adaptive System via MAPE-K")
    print("=" * 70)

    env = UnifiedEnvironment(gym.make("highway-v0"))
    model = load_model("models/best_mapek_model")
    mapek = MAPEKLoop(model, env, history_length=1000, eval_only=True)  # must hold all 1000 episodes

    num_episodes = 1000
    print(f"\n Evaluating for {num_episodes} episodes …")
    print("-" * 70)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = None
        episode_start = np.ones((1,), dtype=bool)  # True at the start of each episode
        done = False
        total_r = 0.0
        steps = 0

        while not done:
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)  # False for all subsequent steps
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += reward
            steps += 1

        metrics = env.get_metrics()
        mapek.monitor(metrics)  # always record
        if (ep + 1) % 50 == 0:  # adapt only every 50 episodes
            mapek.update_model_if_needed()

        if (ep + 1) % 50 == 0 or ep == 0:
            crash_sym = "❌" if metrics["collision_rate"] > 0 else "✅"
            print(
                f"   Ep {ep + 1:4d}/{num_episodes} | Steps: {steps:3d} | "
                f"Reward: {total_r:7.2f} | Crash: {crash_sym} | "
                f"Detect: {metrics['detection_rate']:.3f} | "
                f"FP: {metrics['false_positive_rate']:.3f} | "
                f"Speed: {metrics['avg_speed']:.1f}"
            )

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    plot_performance(mapek.get_history(), save_dir="results")

    best = mapek.get_best_metrics()
    if best:
        print("\n  Best episode metrics:")
        for k, v in best.items():
            print(f"   {k:30s}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    env.close()


if __name__ == "__main__":
    main()

