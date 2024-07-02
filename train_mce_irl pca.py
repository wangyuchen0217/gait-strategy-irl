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

SEED = 42
rng = np.random.default_rng(SEED)
n_bins = 10  # Adjust based on your specific needs
number_of_features = 3  # Adjust based on your specific needs
gamma = 0.99

# Create the environment
exclude_xy = config_data.get("exclude_xy")
env = gym.make('StickInsect-v0-discrete',
               exclude_current_positions_from_observation=exclude_xy,
               max_episode_steps=3000)
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
env.horizon = 3000
env.state_dim = number_of_features
env.action_dim = number_of_features
env.state_space = gym.spaces.Discrete(n_bins ** number_of_features)
env.action_space = gym.spaces.Discrete(n_bins)
env.observation_matrix = np.eye(n_bins ** number_of_features)

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
desired_dimension =24
pca = PCA(n_components=desired_dimension)  # Set the number of components to reduce to
pca_result = pca.fit_transform(scaled_data)

# Convert the result back to a DataFrame for easier handling
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(desired_dimension)])
print("pca_result shape:", pca_result.shape)

# Explained variance to understand how much information is retained
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance by each Principal Component:", explained_variance)
print("Cumulative Explained Variance:", cumulative_variance)

# Plotting the explained variance
plt.figure(figsize=(6, 6))
plt.bar(range(1, desired_dimension + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, desired_dimension + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.title('Explained Variance by Principal Components')
plt.legend(loc='best')
plt.savefig('pca_explained_variance.png')



# Calculate state frequencies (histogram)
# Assuming each dimension is discretized into `n_bins` bins
n_bins = 2
# Discretizer for each principal component
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
discretized_data = discretizer.fit_transform(pca_result)
# Convert discretized data to integer states
discretized_states = np.array(discretized_data, dtype=int)

# Calculate the state frequencies (histogram)
state_counts = np.zeros((n_bins,) * desired_dimension, dtype=int)

for state in discretized_states:
    state_counts[tuple(state)] += 1

# Normalize the histogram to get state occupancy
state_occupancy = state_counts / np.sum(state_counts)
# Flatten the state occupancy for use in IRL
state_occupancy_flat = state_occupancy.flatten()

print("State Occupancy:", state_occupancy_flat)
print("State Occupancy Shape:", state_occupancy_flat.shape)
print("Sum of State Occupancy (should be 1):", np.sum(state_occupancy_flat))

# Optional: Visualize the state occupancy if desired
# plt.figure(figsize=(10, 5))
# plt.bar(range(len(state_occupancy_flat)), state_occupancy_flat, alpha=0.5, align='center')
# plt.xlabel('Discretized States')
# plt.ylabel('Occupancy Probability')
# plt.title('State Occupancy Distribution')
# plt.show()