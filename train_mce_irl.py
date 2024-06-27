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


gamma = 0.99
# Configure the discretizer
n_bins = 10  # Adjust based on your specific needs
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='uniform')

# Assuming observations is already loaded and is your continuous data
discrete_observations = discretizer.fit_transform(observations)
discrete_next_observations = discretizer.transform(next_observations)

# # Format transitions for MCE IRL
# transitions = [{
#     "obs": discrete_observations[t],
#     "acts": actions[t],
#     "next_obs": discrete_next_observations[t],
#     "dones": np.array([False]),
#     "infos": {"time_step": t, "discount": gamma ** t}
# } for t in range(len(observations) - 1)]


n_bins = 10  # Adjust based on your specific needs
number_of_features = 2  # Adjust based on your specific needs
# Assuming a suitable discretization or state identification method
state_occupancy = np.zeros((n_bins ** number_of_features,))  # Adjust size based on discretization

for t in range(len(observations) - 1):
    state_index = np.argmax(discrete_observations[t])  # Assuming already one-hot
    discount = gamma ** t
    state_occupancy[state_index] += discount

# Normalize the occupancy measure
state_occupancy /= np.sum(state_occupancy)

# Pass this directly to MCE IRL
demonstrations = state_occupancy  # Directly as a state-occupancy measure



# Initialize MCE IRL
reward_net = BasicRewardNet(env.observation_space, env.action_space)
mce_irl = MCEIRL(
    env=env,
    demonstrations=demonstrations,
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