# Autonomous Vehicle Control & GPS Spoofing Detection

![Autonomous Vehicle Simulation Header](https://via.placeholder.com/1200x400.png?text=Autonomous+Vehicle+Control+%26+GPS+Spoofing+Detection)

An advanced Reinforcement Learning (RL) framework for autonomous vehicle motion planning, speed control, and real-time GPS spoofing detection. This project implements a **4-Phase Curriculum Learning** strategy to progressively train an agent to navigate dense, dynamic traffic environments while simultaneously identifying and mitigating adversarial spatial telemetry manipulation.

> **Note:** Replace the placeholder images with your actual architecture diagrams, training plots, and evaluation figures.

---

# Table of Contents

- [Autonomous Vehicle Control \& GPS Spoofing Detection](#autonomous-vehicle-control--gps-spoofing-detection)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture \& Curriculum Strategy](#architecture--curriculum-strategy)
  - [Phase 1 — Basic Operational Control](#phase-1--basic-operational-control)
  - [Phase 2 — Dynamic Traffic Adaptation](#phase-2--dynamic-traffic-adaptation)
  - [Phase 3 — High-Density Multi-Agent Interaction](#phase-3--high-density-multi-agent-interaction)
  - [Phase 4 — Stress Testing \& Convergence](#phase-4--stress-testing--convergence)
- [Experimental \& Training Results](#experimental--training-results)
  - [1. Collision Rates](#1-collision-rates)
  - [2. Average Safe Distance](#2-average-safe-distance)
  - [3. Average Speed](#3-average-speed)
  - [4. GPS Spoofing Detection Performance](#4-gps-spoofing-detection-performance)
  - [5. Independent Evaluation Phase Summary](#5-independent-evaluation-phase-summary)
- [Directory Structure](#directory-structure)
- [Installation \& Setup](#installation--setup)
  - [Clone the Repository](#clone-the-repository)
  - [Create a Virtual Environment](#create-a-virtual-environment)
    - [Linux/macOS](#linuxmacos)
    - [Windows](#windows)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Train the Agent](#train-the-agent)
  - [Evaluate a Trained Model](#evaluate-a-trained-model)
- [License](#license)

---

# Overview

Modern autonomous vehicles rely heavily on Global Positioning System (GPS) telemetry for global localization and trajectory planning. However, GPS signals are vulnerable to adversarial spoofing attacks that inject false positional offsets, potentially forcing vehicles into dangerous collisions or off-course maneuvers.

This framework introduces a unified reinforcement learning policy that balances motion planning, speed regulation, and safety while integrating a GPS spoofing detection module capable of identifying anomalous spatial telemetry in real time. Through curriculum learning, the agent gradually transitions from simple driving tasks to complex, high-density traffic scenarios under active spoofing attacks.

---

# Key Features

- **4-Phase Curriculum Learning** using varying episode continuation probabilities ($p_{\text{continue}}$).
- **Integrated GPS Spoofing Detection** with near-zero false-positive performance.
- **Safety-Driven Motion Planning** that maintains adaptive car-following distances greater than 30 m.
- **Stable Speed Regulation** around a target velocity of approximately 20 m/s.
- **High-Density Multi-Agent Training** for improved robustness.
- **Independent Evaluation Framework** with comprehensive performance metrics.

---

# Architecture & Curriculum Strategy

Training directly in highly adversarial traffic environments often causes reinforcement learning policies to collapse before meaningful learning occurs.

To address this challenge, training is divided into **3,000,000 timesteps** across four progressively difficult curriculum phases.

![Curriculum Strategy Diagram](https://via.placeholder.com/800x300.png?text=Curriculum+Learning+Phases+Diagram)

## Phase 1 — Basic Operational Control

**Steps:** 0–500k

**Continuation Probability:** $p_{\text{continue}}=0.98$

- Basic speed regulation
- Lane following
- Safe distance maintenance
- Initial GPS spoofing detector training

---

## Phase 2 — Dynamic Traffic Adaptation

**Steps:** 500k–1.2M

**Continuation Probability:** $p_{\text{continue}}=0.92$

- Increased traffic density
- Longer driving episodes
- Frequent GPS spoofing attacks
- Improved policy robustness

---

## Phase 3 — High-Density Multi-Agent Interaction

**Steps:** 1.2M–2.5M

**Continuation Probability:** $p_{\text{continue}}=0.85$

- Aggressive surrounding vehicles
- Extended spoofing attacks
- Noisy observations
- Complex lane-change decisions

---

## Phase 4 — Stress Testing & Convergence

**Steps:** 2.5M–3.0M

**Continuation Probability:** $p_{\text{continue}}=0.85$

- Maximum environmental complexity
- Final policy convergence
- Near-zero collision frequency
- Near-perfect spoofing detection

---

# Experimental & Training Results

## 1. Collision Rates

Across the four curriculum phases, collision frequency steadily decreased as training progressed. Initial exploration produced a collision rate of approximately **2.43%**, while Phase 4 achieved **0.00%** collisions across nearly all evaluation checkpoints.

![Collision Rate Trend](https://via.placeholder.com/800x400.png?text=Collision+Rate+Over+Training+Steps)

---

## 2. Average Safe Distance

The learned policy progressively increased defensive spacing as environmental complexity grew.

- **Phase 1:** ~21.09 m
- **Phase 3–4:** 29.36–40.07 m

These larger safety margins helped absorb unexpected maneuvers during spoofing attacks.

![Safe Distance Trend](https://via.placeholder.com/800x400.png?text=Average+Safe+Distance+Trend)

---

## 3. Average Speed

Despite increasing environmental difficulty, the policy maintained remarkably stable speed regulation throughout training.

- Target Speed: **≈20.03 m/s**
- Equivalent Speed: **≈72.1 km/h**

Temporary speed reductions occurred only during anomaly mitigation before returning to nominal cruise speed.

![Average Speed Trend](https://via.placeholder.com/800x400.png?text=Average+Speed+Stability)

---

## 4. GPS Spoofing Detection Performance

The integrated detection module demonstrated clear learning convergence.

| Metric | Initial | Final |
|---------|---------|-------|
| Detection Rate | 90.81% | **100.0%** |
| False Positive Rate | 25.03% | **<1.0% (0.00% in Phase 4)** |

![Detection Performance](https://via.placeholder.com/800x400.png?text=Spoofing+Detection+Rate+vs+False+Positive+Rate)

---

## 5. Independent Evaluation Phase Summary

After training, the final policy was evaluated across **1,000 independent simulation episodes**.

| Evaluation Metric | Mean | Std. Dev. | Description |
|-------------------|------|-----------|-------------|
| **Collision Rate** | **0.78%** | 3.34% | Near-zero collisions |
| **Safe Distance** | **32.19 m** | — | Defensive spacing maintained |
| **Average Speed** | **19.97 m/s** | — | Stable cruising velocity |
| **Lane Change Success** | **96.42%** | — | High maneuver success |
| **GPS Spoofing Detection Rate** | **99.80%** | 1.97% | Near-perfect attack detection |
| **False Positive Rate** | **0.37%** | — | Minimal false alarms |
| **Composite Performance Score** | **0.937** | — | Overall performance: **93.68%** |

---

# Directory Structure

```text
.
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs and evaluation metrics
├── src/
│   ├── env/                 # Simulation environment
│   ├── models/              # RL policy and GPS spoofing detector
│   ├── utils/               # Logging and utility functions
│   └── train.py             # Training entry point
├── eval.py                  # Evaluation script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

# Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/your-username/autonomous-gps-spoofing-detection.git

cd autonomous-gps-spoofing-detection
```

## Create a Virtual Environment

```bash
python -m venv venv
```

### Linux/macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train the Agent

Run the complete 4-phase curriculum learning pipeline.

```bash
python src/train.py \
    --total-timesteps 3000000 \
    --log-dir ./logs
```

---

## Evaluate a Trained Model

Evaluate a saved checkpoint across 1,000 simulation episodes.

```bash
python eval.py \
    --weights ./checkpoints/phase4_model.zip \
    --episodes 1000
```

---

# License

This project is distributed under the **MIT License**.

See the `LICENSE` file for additional information.
