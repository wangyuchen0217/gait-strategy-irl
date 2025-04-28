# Gait Strategy Inference via Inverse Reinforcement Learning

This project focuses on modeling and encoding the gait strategies of stick insects using Maximum Entropy Inverse Reinforcement Learning (MaxEnt-IRL). The goal is to infer the underlying reward structures that govern gait pattern selection during walking, based on observed kinematic and sensory information.

Parts of this work will be presented at the:
- **AMAM2025 (The 12th International Symposium on Adaptive Motion of Animals and Machines and 2nd LokoAssist Symposium)**.  

A related journal submission to **Bioinspiration & Biomimetics** is currently under review.

![MaxEnt-IRL framework](configs/MaxEnt-IRL.png)

## Introduction

Locomotion in stick insects involves complex gait strategies that dynamically adapt to environmental feedback. This project applies Maximum Entropy Inverse Reinforcement Learning to uncover the reward structure underlying gait selection, based on a set of observed state variables and discrete gait patterns.

- **States**: Walking dynamics (velocity, acceleration, direction) and antennae sensory information.
- **Actions**: Discrete gait patterns identified from locomotion trajectories (including canonical gaits and noncanonical gaits).
- **Objective**: Learn a reward function that best explains the observed gait behaviors under the maximum entropy principle.


## Dataset

- The dataset includes sequential walking episodes recorded from stick insects.
- Each time step includes:
  - **Velocity**: Instantaneous walking speed.
  - **Acceleration**: Change in velocity.
  - **Direction**: Walking heading direction.
  - **Antennae Information**: Tactile sensory input states.
- Ground-truth gait patterns are assigned as discrete action labels.


## Methods

- **Maximum Entropy Inverse Reinforcement Learning (MaxEnt-IRL)** framework is used to infer the reward function from state-action trajectories.
- Discrete state-action space is constructed by binning continuous variables where necessary.
- Feature matrices are designed to encode walking dynamics and antennae feedback.
- The IRL model optimizes the reward structure to maximize the likelihood of observed expert behaviors under the maximum entropy principle.

Training involves:
- Feature expectation matching.
- Reward weight optimization via gradient ascent.
