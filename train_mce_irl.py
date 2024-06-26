import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
)
from imitation.data import rollout
from imitation.rewards import reward_nets
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
from sklearn.preprocessing import KBinsDiscretizer
from scipy.sparse import dok_matrix

SEED = 42
rng = np.random.default_rng(SEED)

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# Create the environment
exclude_xy = config_data.get("exclude_xy")
env = gym.make('StickInsect-v0',
               exclude_current_positions_from_observation=exclude_xy,
               max_episode_steps=3000)
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# Load the expert dataset
obs_states = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-obs.npy', allow_pickle=True)
actions = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-act.npy', allow_pickle=True)

# Extract observations and "actions" (which are the next observations in this context)
observations = obs_states[0, :-1, 2:] if exclude_xy else obs_states[0, :-1, :] # Exclude the last step to avoid indexing error
actions = actions[0, :-1, :] 
next_observations = obs_states[0, 1:, 2:] if exclude_xy else obs_states[0, 1:, :] # Exclude the first step to avoid indexing error



# Parameters
gamma = 0.99  # Discount factor
n_bins = 5  # Fewer bins to reduce dimensionality

# Initialize discretizer
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

# Fit and transform the observations to discretize them
discretized_observations = discretizer.fit_transform(observations)

# Convert to integers
discretized_observations = discretized_observations.astype(int)

# Assuming a lower-dimensional state space for sparse representation
num_states = n_bins ** min(10, observations.shape[1])  # Limiting to 10 dimensions

# Use a sparse matrix to store occupancy measures
state_occupancy = dok_matrix((num_states, 1))

# Calculate discounted occupancy measures
for t, state_index in enumerate(discretized_observations):
    discount = gamma ** t
    # Calculate flat index from multidimensional index
    flat_index = np.ravel_multi_index(state_index[:10], (n_bins,) * min(10, observations.shape[1]))  # Only use up to 10 dimensions
    state_occupancy[flat_index, 0] += discount

# Convert to dense array and normalize
dense_occupancy = state_occupancy.toarray().flatten()
dense_occupancy /= np.sum(dense_occupancy)

print(dense_occupancy)



# Create transitions with time steps
discount_factor = 0.99
trajectories = []
for t in range(len(observations)):
    trajectory = types.TrajectoryWithRew(
        obs=np.array([observations[t], next_observations[t]]),
        acts=np.array([actions[t]]),
        rews=np.array([0.0]),  # Placeholder rewards
        infos=[{'time_step': t}],  # Adding time step information
        terminal=False  # Set terminal to False for all steps
    )
    trajectories.append(trajectory)

# Correctly flatten trajectories to create transitions with rewards
transitions = rollout.flatten_trajectories_with_rew(trajectories)

# Initialize MCE IRL
reward_net = BasicRewardNet(env.observation_space, env.action_space)
mce_irl = MCEIRL(
    env=env,
    demonstrations=transitions,
    reward_net=reward_net,
    rng=rng,
    discount=0.99,
)


env.seed(SEED)
# Train MCE IRL
mce_irl.train(n_epochs=5000)

# Evaluate the learned policy
reward_after_training, _ = evaluate_policy(mce_irl.policy, env, 10)
print(f"Reward after training: {reward_after_training}")

# save the trained model
torch.save(mce_irl.policy, "trained_policy_mce_irl.pth")