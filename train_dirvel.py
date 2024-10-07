import numpy as np
import pandas as pd
from gridworld import CustomMDP as MDP
import maxent
from maxent import customirl
from maxent import maxentirl
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

def build_transition_matrix_from_indices(data, n_states, n_actions):
    transition_counts = np.zeros((n_states, n_actions, n_states))
    # Iterate over the data and infer the next state from the next tuple
    for i in range(len(data) - 1):
        state, action = data[i]          # Current state and action
        next_state, _ = data[i + 1]      # Infer the next state from the next tuple
        # Increment the count for this transition
        transition_counts[state, action, next_state] += 1
    # Compute the sums of transition counts along axis 2
    counts_sum = np.sum(transition_counts, axis=2, keepdims=True)
    # Safely divide the counts by the sums, where counts_sum != 0
    transition_probabilities = np.divide(
        transition_counts, 
        counts_sum, 
        out=np.zeros_like(transition_counts),  # If division by zero, output 0
        where=counts_sum != 0                 # Only divide where counts_sum is not 0
    )
    return transition_probabilities

# Generate trajectories from the dataset
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

transition_probabilities = build_transition_matrix_from_indices(trajectories[0], n_states, n_actions)
print("Transition probabilities shape: ", transition_probabilities.shape)
print("---------------------------------")

def plot_transition_heatmaps(transition_probabilities):
    plt.figure(figsize=(18, 12))
    for action in range(6):
        plt.subplot(2, 3, action+1)
        sns.heatmap(transition_probabilities[:, action, :], cmap="YlGnBu", annot=False)
        plt.title(f"Transition Probabilities for Action {action+1}")
        plt.xlabel("Next State Index")
        plt.ylabel("Current State Index")
    plt.tight_layout()
    plt.savefig('transition_heatmaps.png')

plot_transition_heatmaps(transition_probabilities)

# Apply MaxEnt IRL
epochs = 200
learning_rate = 0.01
discount = 0.9
test_folder = 'test_folder/flatten_traj/maxent/S33A6-tran/'
n_bin1=n_direction_bins
n_bin2=n_velocity_bins
lable_bin1="Direction Bins"
lable_bin2="Velocity Bins"

# # train irl
# rewards = maxentirl(feature_matrix, mdp.n_actions, discount, transition_probabilities, 
#                                         trajectories, epochs, learning_rate, test_folder)
# #Output the inferred rewards
# print("Inferred Rewards:", rewards.shape)
# # Save the inferred rewards as a CSV file
# np.savetxt(test_folder+'inferred_rewards_maxent_direction.csv', rewards, delimiter=',')


def plot_most_rewarded_action(q_values, lable_bin1, lable_bin2, test_folder):
    # Find the action with the highest Q-value for each state
    most_rewarded_action = np.argmax(q_values, axis=1)
    print("Most rewarded action shape: ", most_rewarded_action.shape)
    # Plot the heatmap (reshaping if the states are grid-like, otherwise just plot)
    plt.figure(figsize=(10, 8))
    sns.heatmap(most_rewarded_action.reshape(n_velocity_bins, n_direction_bins), cmap="YlGnBu", annot=True)
    plt.title("Most Rewarded Action for Each State")
    plt.xlabel(lable_bin1)
    plt.ylabel(lable_bin2)
    plt.savefig(test_folder+'most_rewarded_action_heatmap.png')

def plot_q_table(q_values, test_folder):
    plt.figure(figsize=(10, 8))
    plt.title("Q-Table Heatmap (State-Action Rewards)", fontsize=16)
    plt.xlabel("Actions", fontsize=14)
    plt.ylabel("States", fontsize=14)
    plt.imshow(q_values, cmap='viridis', aspect='auto')
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+"q_table_heatmap.png")

def plot_action_reward_subplots(q_values, n_bin1, n_bin2, n_actions, lable_bin1, lable_bin2, test_folder):
    n_states = n_bin1 * n_bin2
    # Set up the figure and the 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    # Iterate over each action index to create a subplot
    for action_index in range(n_actions):
        # Initialize a grid to store the reward for the specified action per (direction, velocity) pair
        reward_grid = np.zeros((n_bin2, n_bin1))
        # Populate the reward grid based on the reward for the specified action
        for state_index in range(n_states):
            bin2 = state_index // n_bin1
            bin1 = state_index % n_bin1
            # Extract the reward for the specified action
            reward_grid[bin2, bin1] = q_values[state_index, action_index]
        # Plotting the heatmap using imshow in the appropriate subplot
        ax = axes[action_index]
        img = ax.imshow(reward_grid, cmap='viridis', aspect='auto')
        ax.set_title(f"Action {action_index}", fontsize=16)
        # ax.set_xticks(ticks=np.arange(0, n_bin1), labels=np.arange(-20, 5, step=5), fontsize=12)
        # ax.set_yticks(ticks=np.arange(0, n_bin2)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
        ax.set_xlabel(lable_bin1, fontsize=14)
        ax.set_ylabel(lable_bin2, fontsize=14)
    # Add a color bar to the last subplot, shared across all subplots
    # change the ax position
    # cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(test_folder+"action_reward_subplots.png")

def plot_singlestate_action(q_values, n_states, n_bin, lable_bin, test_folder):
    n_actions = q_values.shape[1]
    # Initialize a grid to store the aggregated reward per (direction bin, action)
    reward_grid = np.zeros((n_bin, n_actions))
    # Populate the reward grid by aggregating over velocity bins
    for state_index in range(n_states):
        bin = state_index % n_bin
        # Sum rewards across velocity bins for each direction bin and action
        reward_grid[bin, :] += q_values[state_index, :]
    # # Normalize by the number of velocity bins to get an average if needed
    # reward_grid /= n_vel_bins
    plt.figure(figsize=(10, 8))
    plt.imshow(reward_grid, cmap='viridis', aspect='auto')
    plt.title("Reward Heatmap: "+lable_bin+" vs. Action", fontsize=16)
    # plt.yticks(ticks=np.arange(0, n_bin), labels=np.arange(-20, 5, step=5), fontsize=12)
    plt.xlabel("Actions", fontsize=14)
    plt.ylabel(lable_bin, fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+lable_bin+"_action_reward_heatmap.png")

# evaluate the policy
rewards = np.loadtxt(test_folder+'inferred_rewards_maxent_direction.csv', delimiter=',')
q_values = maxent.find_policy(n_states, rewards, n_actions, discount, transition_probabilities)
print("Q-values shape: ", q_values.shape)
# save the q_values as a CSV file
np.savetxt(test_folder+'q_values_maxent_direction.csv', q_values, delimiter=',')
plot_most_rewarded_action(q_values, lable_bin1, lable_bin2, test_folder)
plot_q_table(q_values, test_folder)
plot_action_reward_subplots(q_values, n_bin1, n_bin2, n_actions, lable_bin1, lable_bin2, test_folder)
plot_singlestate_action(q_values, n_states, n_bin1, lable_bin1, test_folder)
plot_singlestate_action(q_values, n_states, n_bin2, lable_bin2, test_folder)
