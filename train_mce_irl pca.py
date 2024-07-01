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
desired_dimension =3
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

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(scaled_data)

# Reshape data for temporal PCA
# Transpose to make the time dimension as features for PCA
scaled_data_transposed = scaled_data.T

# Apply PCA on the transposed data to reduce temporal dimensions
desired_temporal_dimension = 10  # Number of temporal components to reduce to
pca_temporal = PCA(n_components=desired_temporal_dimension)
pca_temporal_result = pca_temporal.fit_transform(scaled_data_transposed)

# Transpose back to get the reduced time dimension
reduced_temporal_data = pca_temporal_result.T

# Plotting the explained variance for temporal PCA
explained_variance_temporal = pca_temporal.explained_variance_ratio_
cumulative_variance_temporal = np.cumsum(explained_variance_temporal)

print("Explained Variance by each Principal Component:", explained_variance)
print("Cumulative Explained Variance:", cumulative_variance)

# Plotting the explained variance
# plt.figure(figsize=(10, 5))
# plt.bar(range(1, desired_dimension + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
# plt.step(range(1, desired_dimension + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
# plt.xlabel('Principal components')
# plt.ylabel('Explained variance ratio')
# plt.title('Explained Variance by Principal Components')
# plt.legend(loc='best')
# plt.show()

# 2D scatter plot of the first two principal components
plt.figure(figsize=(10, 5))
plt.scatter(pca_df['PC2'], pca_df['PC3'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of First Two Principal Components')
plt.grid(True)
plt.show()

# Plotting the principal components over time
plt.figure(figsize=(15, 8))
for i in range(desired_dimension):
    plt.plot(pca_df[f'PC{i+1}'], label=f'PC{i+1}')
plt.xlabel('Time Steps')
plt.ylabel('Principal Component Values')
plt.title('Principal Components Over Time')
plt.legend()
plt.show()

# Plotting the principal components of temporal data
plt.figure(figsize=(15, 8))
for i in range(desired_temporal_dimension):
    plt.plot(reduced_temporal_data[i], label=f'Temporal PC{i+1}')
plt.xlabel('Features')
plt.ylabel('Temporal Principal Component Values')
plt.title('Temporal Principal Components Over Features')
plt.legend()
plt.show()