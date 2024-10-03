import numpy as np
import pandas as pd
from gridworld import CustomMDP as MDP
import maxent
from maxent import customirl
from maxent import maxentirl
import maxent_gpu
import matplotlib.pyplot as plt
import seaborn as sns
from plot_evaluate import *
import torch

# Load the dataset
data = pd.read_csv('expert_demonstration/expert/CarausiusC00.csv')

# Prepare the MDP
n_velocity_bins = data['Velocity Bin'].nunique()
n_direction_bins = data['Direction Bin'].nunique()
n_gait_categories = data['Gait Category'].nunique()
print("---------------------------------")
print("Velocity bins: ", n_velocity_bins)
print("Direction bins: ", n_direction_bins)
print("Gait categories: ", n_gait_categories)
print("---------------------------------")

mdp = MDP(n_velocity_bins, n_direction_bins, n_gait_categories, discount=0.9)

# Create a feature matrix (n_states, n_dimensions)
n_states = mdp.n_states
n_actions = mdp.n_actions
feature_matrix = np.zeros((n_states, n_velocity_bins + n_direction_bins))
print("Feature matrix shape: ", feature_matrix.shape)

# Populate the feature matrix (one-hot encoding)
for index, row in data.iterrows():
    # set the row index
    state_index = int((row['Velocity Bin']-1) * n_direction_bins + (row['Direction Bin']-1))
    # set the one-hot encoding (column index)
    feature_matrix[state_index, (row['Direction Bin']-1)] = 1
    feature_matrix[state_index, n_velocity_bins + (row['Direction Bin']-1)] = 1

def generate_trajectory(data, n_direction_bins):
    trajectory = []
    for index, row in data.iterrows():
        state_index = int((row['Velocity Bin'] - 1) * n_direction_bins + (row['Direction Bin'] - 1))
        action = int(row['Gait Category'])
        trajectory.append([state_index, action])
    return trajectory

# Generate trajectories from the dataset
'''cut_traj array'''
# t01 = data.iloc[0:2456]
# t02 = data.iloc[2456:3960]
# t03 = data.iloc[3960:5199]
# traj_01 = generate_trajectory(t01, n_direction_bins)
# traj_02 = generate_trajectory(t02, n_direction_bins)
# traj_03 = generate_trajectory(t03, n_direction_bins)
# traj_01 = np.array(traj_01)[:1239, :]
# traj_02 = np.array(traj_02)[:1239, :]
# traj_03 = np.array(traj_03)[:1239, :]
# trajectories = np.array([traj_01, traj_02, traj_03])
# print("Trajectories: ", trajectories.shape)

'''cut_traj list'''
# t01 = data.iloc[0:2456]
# t02 = data.iloc[2456:3960]
# t03 = data.iloc[3960:5199]
# traj_01 = generate_trajectory(t01, n_direction_bins)
# traj_02 = generate_trajectory(t02, n_direction_bins)
# traj_03 = generate_trajectory(t03, n_direction_bins)
# trajectories = [traj_01, traj_02, traj_03]
# print("Trajectories: ", len(trajectories), len(trajectories[2]), len(trajectories[1][0]))

'''flatten_traj'''
trajectories = []
for index, row in data.iterrows():
    state_index = int((row['Velocity Bin']-1) * n_direction_bins + (row['Direction Bin']-1))
    action = int(row['Gait Category'])
    trajectories.append([(state_index, action)])
trajectories = np.array(trajectories)
# reshape the trajectories to (1, len_trajectories, 2)
len_trajectories = trajectories.shape[0]
trajectories = trajectories.reshape(1, len_trajectories, 2)
# # trajectories = trajectories.tolist()
# # print("Trajectories: ", len(trajectories), len(trajectories[0]), len(trajectories[0][0]))

# Set up transition probabilities (for simplicity, we'll assume deterministic transitions here)
# transition_probabilities = np.eye(n_states)[np.newaxis].repeat(mdp.n_actions, axis=0)
# transition_probabilities = np.swapaxes(transition_probabilities, 0, 1)

def build_transition_matrix_from_indices(data, n_states, n_actions):
    """
    Build transition probability matrix from (state_idx, action_idx) data.
    
    :param data: A list of tuples (state_idx, action_idx), where the next state is inferred by the next item.
    :param n_states: Number of states.
    :param n_actions: Number of actions.
    :return: Transition probability matrix of shape (n_states, n_actions, n_states).
    """
    # Initialize the transition counts
    transition_counts = np.zeros((n_states, n_actions, n_states))

    # Iterate over the data and infer the next state from the next tuple
    for i in range(len(data) - 1):
        state, action = data[i]          # Current state and action
        next_state, _ = data[i + 1]      # Infer the next state from the next tuple
        
        # Increment the count for this transition
        transition_counts[state, action, next_state] += 1

    # Normalize the counts to get probabilities
    transition_probabilities = transition_counts / np.sum(transition_counts, axis=2, keepdims=True)

    # Handle cases where no transitions were recorded for some state-action pairs
    transition_probabilities = np.nan_to_num(transition_probabilities)

    return transition_probabilities

transition_probabilities = build_transition_matrix_from_indices(trajectories[0], n_states, n_actions)
print("Transition probabilities shape: ", transition_probabilities.shape)
print("---------------------------------")

# Set transition probabilities in MDP
mdp.set_transition_probabilities(transition_probabilities)

# Apply MaxEnt IRL
epochs = 100
learning_rate = 0.01
discount = 0.9
# rewards = customirl(feature_matrix, mdp.n_actions, mdp.discount, transition_probabilities, trajectories, epochs, learning_rate)
# rewards = maxentirl(feature_matrix, mdp.n_actions, discount, 
#                      transition_probabilities, trajectories, epochs, learning_rate)

# #Output the inferred rewards
# print("Inferred Rewards:", rewards.shape)
# print(rewards)
# # Save the inferred rewards as a CSV file
# np.savetxt('inferred_rewards_maxent_direction.csv', rewards, delimiter=',')

rewards = np.loadtxt('test_folder/flatten_traj/maxent/S33A6/inferred_rewards_maxent_direction.csv', delimiter=',')
q_values = maxent.find_policy(n_states, rewards, n_actions, discount, transition_probabilities)
print("Q-values shape: ", q_values.shape)
# save the q_values as a CSV file
np.savetxt('q_values_maxent_direction.csv', q_values, delimiter=',')

def plot_most_rewarded_action(q_values, n_states):
    """
    Plot heatmap of the most rewarded action for each state based on Q-values.
    
    :param q_values: (n_states, n_actions) matrix of Q-values for each state-action pair.
    :param n_states: Number of states.
    """
    # Find the action with the highest Q-value for each state
    most_rewarded_action = np.argmax(q_values, axis=1)
    print("Most rewarded action shape: ", most_rewarded_action.shape)

    # Plot the heatmap (reshaping if the states are grid-like, otherwise just plot)
    plt.figure(figsize=(10, 6))
    sns.heatmap(most_rewarded_action.reshape(n_velocity_bins, n_direction_bins), cmap="YlGnBu", annot=True)
    plt.title("Most Rewarded Action for Each State")
    plt.xlabel("State Index")
    plt.ylabel("State Index")
    plt.show()

plot_most_rewarded_action(q_values, n_states)

# Evaluate the inferred rewards
# rewards = np.loadtxt('inferred_rewards.csv', delimiter=',')
# plot_grid_based_rewards(rewards, n_direction_bins, n_velocity_bins)
# visualize_rewards_heatmap(rewards, n_states, mdp.n_actions)
# plot_most_rewarded_action_heatmap(rewards, n_direction_bins=5, n_vel_bins=28)
# plot_action_reward_subplots(rewards, n_direction_bins=5, n_vel_bins=28, n_actions=6)
# plot_velocity_action_reward_heatmap(rewards, n_direction_bins=5, n_vel_bins=28)
# plot_direction_action_reward_heatmap(rewards, n_direction_bins=5, n_vel_bins=28)