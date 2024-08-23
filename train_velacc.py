import numpy as np
import pandas as pd
from gridworld import CustomMDP as MDP
from maxent import customirl as irl
import matplotlib.pyplot as plt
import seaborn as sns
from plot_evaluate import *

# Load the dataset
data = pd.read_csv('expert_demonstration/expert/CarausiusC00.csv')

# Prepare the MDP
n_velocity_bins = data['Velocity Bin'].nunique()
n_acceleration_bins = data['Acceleration Bin'].nunique()
n_gait_categories = data['Gait Category'].nunique()
print("---------------------------------")
print("Velocity bins: ", n_velocity_bins)
print("Acceleration bins: ", n_acceleration_bins)
print("Gait categories: ", n_gait_categories)
print("---------------------------------")

mdp = MDP(n_velocity_bins, n_acceleration_bins, n_gait_categories, discount=0.9)

# Create a feature matrix (n_states, n_dimensions)
n_states = mdp.n_states
feature_matrix = np.zeros((n_states, n_velocity_bins + n_acceleration_bins))
print("Feature matrix shape: ", feature_matrix.shape)

# Populate the feature matrix (one-hot encoding)
for index, row in data.iterrows():
    # set the row index
    state_index = int((row['Velocity Bin']-1) * n_acceleration_bins + (row['Acceleration Bin']-1))
    # set the one-hot encoding (column index)
    if state_index == 378:
        print(row['Acceleration Bin'])
    feature_matrix[state_index, (row['Acceleration Bin']-1)] = 1
    feature_matrix[state_index, n_velocity_bins + (row['Acceleration Bin']-1)] = 1

# Generate trajectories from the dataset
trajectories = []
for index, row in data.iterrows():
    state_index = int((row['Velocity Bin']-1) * n_acceleration_bins + (row['Acceleration Bin']-1))
    action = int(row['Gait Category'])
    trajectories.append([(state_index, action)])

# Set up transition probabilities (for simplicity, we'll assume deterministic transitions here)
transition_probabilities = np.eye(n_states)[np.newaxis].repeat(mdp.n_actions, axis=0)
transition_probabilities = np.swapaxes(transition_probabilities, 0, 1)
print("Transition probabilities shape: ", transition_probabilities.shape)
print("---------------------------------")

# Set transition probabilities in MDP
mdp.set_transition_probabilities(transition_probabilities)

# Apply MaxEnt IRL
epochs = 100
learning_rate = 0.01
# rewards = irl(feature_matrix, mdp.n_actions, mdp.discount, transition_probabilities, trajectories, epochs, learning_rate)

# # Output the inferred rewards
# print("Inferred Rewards:", rewards.shape)
# print(rewards)
# # Save the inferred rewards as a CSV file
# np.savetxt('inferred_rewards.csv', rewards, delimiter=',')

rewards = np.loadtxt('inferred_rewards.csv', delimiter=',')

plot_grid_based_rewards(rewards, n_acceleration_bins, n_velocity_bins)
visualize_rewards_heatmap(rewards, n_states, mdp.n_actions)
plot_most_rewarded_action_heatmap(rewards, n_acceleration_bins=21, n_vel_bins=28)
plot_action_reward_subplots(rewards, n_acceleration_bins=21, n_vel_bins=28, n_actions=6)
plot_velocity_action_reward_heatmap(rewards, n_acceleration_bins=21, n_vel_bins=28)
plot_acceleration_action_reward_heatmap(rewards, n_acceleration_bins=21, n_vel_bins=28)