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
env.state_space = gym.spaces.Discrete(n_bins )
env.action_space = gym.spaces.Discrete(n_bins)
env.observation_matrix = np.eye(n_bins)

# Load the expert dataset
obs_states = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-obs.npy', allow_pickle=True)
actions = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-act.npy', allow_pickle=True)


# Extract qpos and qvel data
qpos_data = obs_states[0, :, 2:-30]  # Adjust indices based on actual data layout
qvel_data = obs_states[0, :, -30:]  # Adjust indices based on actual data layout

# Setup PCA
pca_qpos = PCA(n_components=number_of_features)  # Reduce to 10 principal components for qpos
pca_qvel = PCA(n_components=number_of_features)  # Reduce to 10 principal components for qvel

# Fit PCA on flattened data assuming the first dimension is the batch dimension
reduced_qpos = pca_qpos.fit_transform(qpos_data)
reduced_qvel = pca_qvel.fit_transform(qvel_data)

# Combine reduced qpos and qvel data
reduced_data = np.hstack((reduced_qpos, reduced_qvel))

# Use new_observations for training or simulation
print("Transformed Observations Shape:", reduced_data.shape)

num_bins = 10
state_occupancy = np.zeros(num_bins)

# Example of discretizing and counting occupancy (very simplistic and for illustrative purposes)
# Normally, you would have exact state definitions to increment these counts correctly
for observation in reduced_data:
    index = int(np.sum(observation) * num_bins / np.max(reduced_data)) % num_bins
    state_occupancy[index] += 1

# Normalize to form a probability distribution (or keep as counts if that's what the method expects)
state_occupancy /= np.sum(state_occupancy)


class CustomRewardNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Example layer
        self.layer = torch.nn.Linear(10, 1)  # Adjust dimensions according to your input size

    def forward(self, obs, action=None, *args):
        if action is not None:
            x = torch.cat((obs, action), dim=-1)  # Concatenate action to observation if not None
        else:
            x = obs
        return self.layer(x)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype  # Returns the dtype of the first parameter
    
    @property
    def device(self):
        return next(self.parameters()).device  # Returns the device of the first parameter

    
reward_net = CustomRewardNet()


# Initialize MCE IRL
# reward_net = BasicRewardNet(env.observation_space, env.action_space)
mce_irl = MCEIRL(
    env=env,
    demonstrations=state_occupancy,
    reward_net=reward_net,
    rng=rng,
    discount=0.99,
)


env.seed(SEED)
# Train MCE IRL
mce_irl.train()

# Evaluate the learned policy
reward_after_training, _ = evaluate_policy(mce_irl.policy, env, 10)
print(f"Reward after training: {reward_after_training}")

# save the trained model
torch.save(mce_irl.policy, "trained_policy_mce_irl.pth")