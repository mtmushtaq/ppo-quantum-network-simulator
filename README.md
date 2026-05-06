# PPO-Based Entanglement Scheduler for LEO Satellite Quantum Networks

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-MaskablePPO-purple)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://arxiv.org/)

---

## Overview

This repository contains the full implementation of the framework presented in:

> **"PPO-Based Entanglement Scheduling for LEO Satellite Quantum Networks:
> A Physics-Accurate MDP Formulation"**
> Muhammad Tauseef Mushtaq, Vito Guida, Nicola Cordeschi
> Department of Electrical and Information Engineering, Politecnico di Bari

We address the problem of entanglement request scheduling in Low Earth Orbit
(LEO) satellite quantum networks using a physics-grounded Markov Decision
Process (MDP) solved by Proximal Policy Optimization (PPO) with dynamic
action masking. The environment integrates a complete physical-layer model —
free-space diffraction, atmospheric attenuation, iterative entanglement
purification, and quantum memory decoherence — running on real Starlink
orbital ephemeris data.

Against a fidelity-greedy deterministic baseline, the trained PPO agent achieves:

- **↑ 84% higher throughput** (served requests per second) at TTL = 1 s
- **↓ 6.3× fewer expired requests** at TTL = 1 s
- **Stable 110–122 req/s** across all TTL regimes (1 s – 5 s)
- **> 99.9% success rate** and **~94% average fidelity** in all configurations

The Greedy baseline *degrades* as TTL increases (queue-clogging),
while PPO exploits longer coherence windows — demonstrating that
AI-driven scheduling is a necessary complement to hardware improvements
in future quantum network infrastructure.

---

## Key Features

- **Physics-accurate environment**: free-space Friis loss, atmospheric
  transmittance, background-photon fidelity degradation, iterative
  purification (up to 3 rounds), and BSM-based entanglement swapping.

- **Micro-step MDP architecture**: the 22 ms physical clock is frozen
  while three concurrent entanglement requests are evaluated sequentially,
  reducing the action space from O(|S|³) to O(|S|) per step without
  sacrificing temporal consistency.

- **Dynamic action masking**: geometric Line-of-Sight (LoS) feasibility
  and per-macro-step satellite mutex locks are enforced as hard constraints
  at every decision step via `MaskablePPO`, eliminating invalid allocations
  entirely.

- **Real orbital data**: Starlink TLE sets are fetched from
  Celestrak/Spacetrack and propagated at 22 ms resolution to produce
  per-timestep slant range, elevation angle, and GS visibility masks.

- **Deterministic Greedy baseline**: identical physics model, fidelity-optimal
  myopic scheduler — a fair and transparent comparison.

- **Multi-objective reward**: throughput, average post-purification fidelity,
  and TTL-expiration penalty with configurable weights (α, β, δ).

---

## Repository Structure
