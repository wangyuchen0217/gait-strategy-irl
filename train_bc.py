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
from stable_baselines3.common.monitor import Monitor
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
import torch
from pykalman import KalmanFilter

SEED = 42

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# Create the environment
exclude_xy = config_data.get("exclude_xy")
env = gym.make('StickInsect-v0',
               exclude_current_positions_from_observation=exclude_xy,
               max_episode_steps=3000)
env = Monitor(env) 
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# Load the expert dataset
obs_states = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-obs.npy', allow_pickle=True)
actions = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-act.npy', allow_pickle=True)

# Extract observations and "actions" (which are the next observations in this context)
observations = obs_states[0, :-1, 2:] if exclude_xy else obs_states[0, :-1, :] # Exclude the last step to avoid indexing error
actions = actions[0, :-1, :] 

next_observations = obs_states[0, 1:, 2:] if exclude_xy else obs_states[0, 1:, :] # Exclude the first step to avoid indexing error

dones = np.zeros(len(observations), dtype=bool)
# dones[-1] = True  # Mark the last timestep as terminal

# transit the data to types.Transitions
transitions = types.Transitions(
    obs=observations,
    acts=actions,
    next_obs=next_observations,
    dones=dones,
    infos=[{} for _ in range(len(observations))]
)

# Create the BC trainer
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=np.random.default_rng(SEED)
)

reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
bc_trainer.train(n_epochs=5000)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)

print(f"Reward before training: {reward_before_training}")
print(f"Reward after training: {reward_after_training}")

# save the trained model
torch.save(bc_trainer, "trained_policy_bc.pth")