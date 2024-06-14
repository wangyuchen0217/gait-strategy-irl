import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.data import types
import envs

import yaml
import matplotlib.pyplot as plt
import logging

SEED = 42

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# Create the environment
exclude_xy = config_data.get("exclude_xy")
env = gym.make('StickInsect-v0',  exclude_current_positions_from_observation=exclude_xy)
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# Load the expert dataset
expert = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12.npy', allow_pickle=True)

# Extract observations and "actions" (which are the next observations in this context)
# Avoid using the last transition because it lacks a valid next observation
observations = expert[0, :-1, :-48] if exclude_xy else expert[0, :-2, :]
actions = expert[0, 1:-1, -48:]   # Adjusted to avoid the last transition
next_observations = expert[0, 2:, 2:] if exclude_xy else expert[0, 2:, :]

dones = np.zeros(len(observations), dtype=bool)
# dones[-1] = True  # Mark the last timestep as terminal

transitions = {
    'obs': observations,
    'acts': actions,
    'next_obs': next_observations,
    'dones': np.zeros(len(observations), dtype=bool)
}


# Create the BC trainer
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=np.random.default_rng(SEED)
)

reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before training: {reward_before_training}")

bc_trainer.train(n_epochs=1)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")