import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.mce_irl import MCEIRL
from imitation.algorithms.mce_irl import squeeze_r

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)
exclude_xy = config_data['exclude_xy']


# Load the expert dataset
obs_states = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-obs.npy', allow_pickle=True)
actions = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-act.npy', allow_pickle=True)

# Extract observations and "actions" (which are the next observations in this context)
observations = obs_states[0, :-1, 2:] if exclude_xy else obs_states[0, :-1, :] # Exclude the last step to avoid indexing error
actions = actions[0, :-1, :] 
next_observations = obs_states[0, 1:, 2:] if exclude_xy else obs_states[0, 1:, :] # Exclude the first step to avoid indexing error

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(observations)

# Apply PCA
pca_dimension = config_data['irl']['pca_dimension']
pca = PCA(n_components=pca_dimension)  # Set the number of components to reduce to
pca_result = pca.fit_transform(scaled_data)
# Convert the result back to a DataFrame for easier handling
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_dimension)])
print("pca_result shape:", pca_result.shape)

# Explained variance to understand how much information is retained
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print("Explained Variance by each Principal Component:", explained_variance)
print("Cumulative Explained Variance:", cumulative_variance)

# Plotting the explained variance
plt.figure(figsize=(6, 6))
plt.bar(range(1, pca_dimension + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, pca_dimension + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.title('Explained Variance by Principal Components')
plt.legend(loc='best')
plt.savefig('pca_explained_variance.png')



# # Calculate state frequencies (histogram)
# # Assuming each dimension is discretized into `n_bins` bins
# n_bins = config_data['irl']['n_bins']
# # Discretizer for each principal component
# discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
# discretized_data = discretizer.fit_transform(pca_result)
# # Convert discretized data to integer states
# discretized_states = np.array(discretized_data, dtype=int)

# # Calculate the state frequencies (histogram)
# state_counts = np.zeros((n_bins,) * pca_dimension, dtype=int)
# for state in discretized_states:
#     state_counts[tuple(state)] += 1
# # Normalize the histogram to get state occupancy
# state_occupancy = state_counts / np.sum(state_counts)
# # Flatten the state occupancy for use in IRL
# state_occupancy_flat = state_occupancy.flatten()

# print("Discretized States Shape:", discretized_states.shape)
# print("State_counts shape:", state_counts.shape)
# print("State Occupancy:", state_occupancy_flat)
# print("State Occupancy Shape:", state_occupancy_flat.shape)
# print("Sum of State Occupancy (should be 1):", np.sum(state_occupancy_flat))


# SEED = config_data['env']['seed']
# horizon = config_data['env']['horizon']
# rng = np.random.default_rng(SEED)
# state_dim = pca_dimension
# action_dim = len(actions[0])

# # Create the environment
# env = gym.make('StickInsect-v0-disc',
#                exclude_current_positions_from_observation=exclude_xy,
#                max_episode_steps=horizon)
# env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
# env.horizon = horizon
# env.state_dim = n_bins ** state_dim
# env.action_dim = action_dim
# env.state_space = gym.spaces.Discrete(n_bins ** state_dim)
# env.action_space = gym.spaces.Discrete(action_dim)
# env.observation_matrix = np.eye(n_bins ** state_dim)
# env.transition_matrix = np.zeros((n_bins ** state_dim, action_dim, n_bins ** state_dim))
# env.initial_state_dist = np.zeros(n_bins ** state_dim)

# # Initialize reward network
# reward_net = BasicRewardNet(
#     observation_space=env.state_space,
#     action_space=env.action_space,
#     use_state=True,
#     use_action=False,
#     use_next_state=False,
#     use_done=False,
# )

# # Initialize MCE-IRL algorithm
# mce_irl = MCEIRL(
#     demonstrations=state_occupancy_flat,  # Provide state occupancy as demonstrations
#     env=env,
#     reward_net=reward_net,
#     rng=rng,
# )

# env.seed(SEED)
# mce_irl.train()

# # save the trained model
# torch.save(mce_irl.policy, "trained_policy_mce_irl.pth")


