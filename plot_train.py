import matplotlib.pyplot as plt
import numpy as np


def plot_grid_based_rewards(rewards, n_direction_bins, n_vel_bins, epoch):
    # Assuming rewards are aggregated over actions for each state
    state_rewards = rewards.sum(axis=1).reshape((n_vel_bins, n_direction_bins))
    plt.figure(figsize=(10, 8))
    plt.imshow(state_rewards, cmap='viridis_r', aspect='auto')
    plt.title("Grid-Based Reward Heatmap", fontsize=16)
    plt.xlabel("Direction Bins", fontsize=14)
    plt.ylabel("Velocity Bins", fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig("reward_heatmap_"+ epoch +".png")

# def plot_grid_based_rewards(rewards, n_acceleration_bins, n_vel_bins, epoch):
#     # Assuming rewards are aggregated over actions for each state
#     state_rewards = rewards.sum(axis=1).reshape((n_vel_bins, n_acceleration_bins))
#     plt.figure(figsize=(10, 8))
#     plt.imshow(state_rewards, cmap='viridis', aspect='auto')
#     plt.title("Grid-Based Reward Heatmap")
#     plt.xlabel("Acceleration Bins")
#     plt.ylabel("Velocity Bins")
#     plt.colorbar(label='Reward Value')
#     plt.savefig("reward_heatmap_"+ epoch +".png")

def plot_most_rewarded_action_heatmap(rewards, n_direction_bins, n_vel_bins, epoch):
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
    plt.savefig("action_heatmap_"+ epoch +".png")

