import numpy as np
import pandas as pd
from gridworld import CustomMDP as MDP
from maxent import customirl as irl
import matplotlib.pyplot as plt
import seaborn as sns

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
feature_matrix = np.zeros((n_states, n_velocity_bins + n_direction_bins))
print("Feature matrix shape: ", feature_matrix.shape)

# Populate the feature matrix (one-hot encoding)
for index, row in data.iterrows():
    # set the row index
    state_index = int((row['Velocity Bin']-1) * n_direction_bins + (row['Direction Bin']-1))
    # set the one-hot encoding (column index)
    feature_matrix[state_index, (row['Direction Bin']-1)] = 1
    feature_matrix[state_index, n_velocity_bins + (row['Direction Bin']-1)] = 1

# Generate trajectories from the dataset
trajectories = []
for index, row in data.iterrows():
    state_index = int((row['Velocity Bin']-1) * n_direction_bins + (row['Direction Bin']-1))
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

def plot_most_rewarded_action_heatmap(rewards, n_direction_bins, n_vel_bins):
    """
    Creates a heatmap where each cell shows the most rewarded action for each (direction bin, velocity bin) pair.
    
    :param rewards: Reward matrix of shape (n_states, n_actions).
    :param n_direction_bins: The number of direction bins.
    :param n_vel_bins: The number of velocity bins.
    """
    n_states = n_direction_bins * n_vel_bins

    # Initialize a grid to store the most rewarded action per (direction, velocity) pair
    action_grid = np.zeros((n_vel_bins, n_direction_bins), dtype=int)

    # Determine the most rewarded action for each (direction, velocity) pair
    for state_index in range(n_states):
        velocity_bin = state_index // n_direction_bins
        direction_bin = state_index % n_direction_bins

        # Find the action with the highest reward for this state
        most_rewarded_action = np.argmax(rewards[state_index, :])
        action_grid[velocity_bin, direction_bin] = most_rewarded_action

    # Create a discrete color map for actions
    n_actions = rewards.shape[1]
    cmap = plt.cm.get_cmap('YlGnBu', n_actions)  # Using 'tab20' for up to 20 actions

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    img = plt.imshow(action_grid, cmap=cmap, aspect='auto')
    plt.title("Most Rewarded Action Heatmap")
    plt.xlabel("Direction Bins")
    plt.ylabel("Velocity Bins")

    # Create a color bar with action labels
    cbar = plt.colorbar(img, ticks=np.arange(np.max(action_grid) + 1))
    cbar.set_label('Actions')
    cbar.set_ticks(np.arange(np.max(action_grid) + 1) + 0.5)
    cbar.set_ticklabels([f"Action {i}" for i in range(np.max(action_grid) + 1)])

    plt.show()

# Example usage:
# Assuming rewards is a (140, 25) array and n_direction_bins=8, n_vel_bins=25
plot_most_rewarded_action_heatmap(rewards, n_direction_bins=5, n_vel_bins=28)

