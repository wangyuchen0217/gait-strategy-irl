import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# def plot_grid_based_rewards(rewards, n_direction_bins, n_vel_bins):
#     # Assuming rewards are aggregated over actions for each state
#     state_rewards = rewards.sum(axis=1).reshape((n_vel_bins, n_direction_bins))
#     plt.figure(figsize=(10, 8))
#     #plt.imshow(state_rewards, cmap='viridis', aspect='auto')
#     plt.imshow(state_rewards, cmap='viridis_r', aspect='auto')  # Use 'viridis_r' for reversed colormap
#     plt.title("Grid-Based Reward Heatmap", fontsize=16)
#     plt.xticks(ticks=np.arange(0, n_direction_bins), labels=np.arange(-20, 5, step=5), fontsize=12)
#     plt.yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
#     plt.xlabel("Direction Bins", fontsize=14)
#     plt.ylabel("Velocity Bins", fontsize=14)
#     plt.colorbar(label='Reward Value')
#     plt.savefig("grid_base_reward_heatmap.png")

# def visualize_rewards_heatmap(rewards, n_states, n_actions):
#     # Option 1: Visualize the full reward matrix as a heatmap
#     plt.figure(figsize=(10, 8))
#     plt.title("Reward Function Heatmap (State-Action Rewards)", fontsize=16)
#     plt.xlabel("Actions", fontsize=14)
#     plt.ylabel("States", fontsize=14)
#     #plt.imshow(rewards, cmap='viridis', aspect='auto')
#     plt.imshow(rewards, cmap='viridis_r', aspect='auto')
#     plt.colorbar(label='Reward Value')
#     plt.savefig("state_action_reward_heatmap.png")

# def plot_most_rewarded_action_heatmap(rewards, n_direction_bins, n_vel_bins):
#     """
#     Creates a heatmap where each cell shows the most rewarded action for each (direction bin, velocity bin) pair.
    
#     :param rewards: Reward matrix of shape (n_states, n_actions).
#     :param n_direction_bins: The number of direction bins.
#     :param n_vel_bins: The number of velocity bins.
#     """
#     n_states = n_direction_bins * n_vel_bins

#     # Initialize a grid to store the most rewarded action per (direction, velocity) pair
#     action_grid = np.zeros((n_vel_bins, n_direction_bins), dtype=int)

#     # Determine the most rewarded action for each (direction, velocity) pair
#     for state_index in range(n_states):
#         velocity_bin = state_index // n_direction_bins
#         direction_bin = state_index % n_direction_bins

#         # Find the action with the highest reward for this state
#         most_rewarded_action = np.argmax(rewards[state_index, :])
#         action_grid[velocity_bin, direction_bin] = most_rewarded_action

#     # Create a discrete color map for actions
#     n_actions = rewards.shape[1]
#     cmap = plt.cm.get_cmap('YlGnBu', n_actions)  # Using 'tab20' for up to 20 actions

#     # Plotting the heatmap
#     plt.figure(figsize=(10, 8))
#     img = plt.imshow(action_grid, cmap=cmap, aspect='auto')
#     plt.title("Most Rewarded Action Heatmap", fontsize=16)
#     plt.xticks(ticks=np.arange(0, n_direction_bins), labels=np.arange(-20, 5, step=5), fontsize=12)
#     plt.yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
#     plt.xlabel("Direction Bins", fontsize=14)
#     plt.ylabel("Velocity Bins", fontsize=14)

#     # Create a color bar with action labels
#     cbar = plt.colorbar(img, ticks=np.arange(np.max(action_grid) + 1))
#     cbar.set_label('Actions')
#     cbar.set_ticks(np.arange(np.max(action_grid) + 1) + 0.5)
#     cbar.set_ticklabels([f"Action {i}" for i in range(np.max(action_grid) + 1)])

#     plt.savefig("most_rewarded_action_heatmap.png")

# def plot_action_reward_subplots(rewards, n_direction_bins, n_vel_bins, n_actions):
#     """
#     Creates a 2x3 grid of subplots showing the reward for each action across the velocity-direction grid.
    
#     :param rewards: Reward matrix of shape (n_states, n_actions).
#     :param n_direction_bins: The number of direction bins.
#     :param n_vel_bins: The number of velocity bins.
#     :param n_actions: The number of actions to visualize.
#     """
#     n_states = n_direction_bins * n_vel_bins

#     # Set up the figure and the 2x3 subplot grid
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#     axes = axes.flatten()

#     # Iterate over each action index to create a subplot
#     for action_index in range(n_actions):
#         # Initialize a grid to store the reward for the specified action per (direction, velocity) pair
#         reward_grid = np.zeros((n_vel_bins, n_direction_bins))

#         # Populate the reward grid based on the reward for the specified action
#         for state_index in range(n_states):
#             velocity_bin = state_index // n_direction_bins
#             direction_bin = state_index % n_direction_bins

#             # Extract the reward for the specified action
#             reward_grid[velocity_bin, direction_bin] = rewards[state_index, action_index]

#         # Plotting the heatmap using imshow in the appropriate subplot
#         ax = axes[action_index]
#         img = ax.imshow(reward_grid, cmap='viridis_r', aspect='auto')
#         ax.set_title(f"Action {action_index}", fontsize=16)
#         ax.set_xticks(ticks=np.arange(0, n_direction_bins), labels=np.arange(-20, 5, step=5), fontsize=12)
#         ax.set_yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
#         ax.set_xlabel("Direction Bins", fontsize=14)
#         ax.set_ylabel("Velocity Bins", fontsize=14)

#     # Add a color bar to the last subplot, shared across all subplots
#     # change the ax position
#     # cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#     fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
#     plt.tight_layout(rect=[0, 0, 0.88, 1])
#     plt.savefig("action_reward_subplots.png")

# def plot_velocity_action_reward_heatmap(rewards, n_direction_bins, n_vel_bins):
#     """
#     Creates a heatmap showing the reward distribution across velocity bins and actions.
    
#     :param rewards: Reward matrix of shape (n_states, n_actions).
#     :param n_direction_bins: The number of direction bins.
#     :param n_vel_bins: The number of velocity bins.
#     """
#     n_states = n_direction_bins * n_vel_bins
#     n_actions = rewards.shape[1]

#     # Initialize a grid to store the aggregated reward per (velocity bin, action)
#     reward_grid = np.zeros((n_vel_bins, n_actions))

#     # Populate the reward grid by aggregating over direction bins
#     for state_index in range(n_states):
#         velocity_bin = state_index // n_direction_bins
#         # Sum rewards across direction bins for each velocity bin and action
#         reward_grid[velocity_bin, :] += rewards[state_index, :]

#     # Normalize by the number of direction bins to get an average if needed
#     reward_grid /= n_direction_bins

#     # Plotting the heatmap using imshow
#     plt.figure(figsize=(10, 8))
#     plt.imshow(reward_grid, cmap='viridis_r', aspect='auto')
#     plt.title("Reward Heatmap: Velocity vs. Action", fontsize=16)
#     plt.yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
#     plt.xlabel("Actions", fontsize=14)
#     plt.ylabel("Velocity Bins", fontsize=14)
#     plt.colorbar(label='Reward Value')
#     plt.savefig("velocity_action_reward_heatmap.png")

# def plot_direction_action_reward_heatmap(rewards, n_direction_bins, n_vel_bins):
#     """
#     Creates a heatmap showing the reward distribution across direction bins and actions.
    
#     :param rewards: Reward matrix of shape (n_states, n_actions).
#     :param n_direction_bins: The number of direction bins.
#     :param n_vel_bins: The number of velocity bins.
#     """
#     n_states = n_direction_bins * n_vel_bins
#     n_actions = rewards.shape[1]

#     # Initialize a grid to store the aggregated reward per (direction bin, action)
#     reward_grid = np.zeros((n_direction_bins, n_actions))

#     # Populate the reward grid by aggregating over velocity bins
#     for state_index in range(n_states):
#         direction_bin = state_index % n_direction_bins
#         # Sum rewards across velocity bins for each direction bin and action
#         reward_grid[direction_bin, :] += rewards[state_index, :]

#     # Normalize by the number of velocity bins to get an average if needed
#     reward_grid /= n_vel_bins

#     # Plotting the heatmap using imshow
#     plt.figure(figsize=(10, 8))
#     plt.imshow(reward_grid, cmap='viridis_r', aspect='auto')
#     plt.title("Reward Heatmap: Direction vs. Action", fontsize=16)
#     plt.yticks(ticks=np.arange(0, n_direction_bins), labels=np.arange(-20, 5, step=5), fontsize=12)
#     plt.xlabel("Actions", fontsize=14)
#     plt.ylabel("Direction Bins", fontsize=14)
#     plt.colorbar(label='Reward Value')
#     plt.savefig("direction_action_reward_heatmap.png")

''''''

def plot_grid_based_rewards(rewards, n_acceleration_bins, n_vel_bins):
    # Assuming rewards are aggregated over actions for each state
    state_rewards = rewards.sum(axis=1).reshape((n_vel_bins, n_acceleration_bins))
    plt.figure(figsize=(10, 8))
    plt.imshow(state_rewards, cmap='viridis_r', aspect='auto')
    plt.title("Grid-Based Reward Heatmap", fontsize=16)
    plt.xticks(ticks=np.arange(0, n_acceleration_bins)[::3], labels=np.arange(-3000, 2250, step=250)[::3], minor=False, fontsize=12)
    plt.yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
    plt.xlabel("Acceleration Bins", fontsize=14)
    plt.ylabel("Velocity Bins", fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig("grid_base_reward_heatmap.png")

def visualize_rewards_heatmap(rewards, n_states, n_actions):
    # Option 1: Visualize the full reward matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.title("Reward Function Heatmap (State-Action Rewards)", fontsize=16)
    plt.xlabel("Actions", fontsize=14)
    plt.ylabel("States", fontsize=14)
    plt.imshow(rewards, cmap='viridis_r', aspect='auto')
    plt.colorbar(label='Reward Value')
    plt.savefig("state_action_reward_heatmap.png")

def plot_most_rewarded_action_heatmap(rewards, n_acceleration_bins, n_vel_bins):
    n_states = n_acceleration_bins * n_vel_bins

    # Initialize a grid to store the most rewarded action per (direction, velocity) pair
    action_grid = np.zeros((n_vel_bins, n_acceleration_bins), dtype=int)

    # Determine the most rewarded action for each (direction, velocity) pair
    for state_index in range(n_states):
        velocity_bin = state_index // n_acceleration_bins
        direction_bin = state_index % n_acceleration_bins

        # Find the action with the highest reward for this state
        most_rewarded_action = np.argmax(rewards[state_index, :])
        action_grid[velocity_bin, direction_bin] = most_rewarded_action

    # Create a discrete color map for actions
    n_actions = rewards.shape[1]
    cmap = plt.cm.get_cmap('YlGnBu', n_actions)  # Using 'tab20' for up to 20 actions

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    img = plt.imshow(action_grid, cmap=cmap, aspect='auto')
    plt.title("Most Rewarded Action Heatmap", fontsize=16)
    plt.xticks(ticks=np.arange(0, n_acceleration_bins)[::3], labels=np.arange(-3000, 2250, step=250)[::3], minor=False, fontsize=12)
    plt.yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
    plt.xlabel("Acceleration Bins", fontsize=14)
    plt.ylabel("Velocity Bins", fontsize=14)

    # Create a color bar with action labels
    cbar = plt.colorbar(img, ticks=np.arange(np.max(action_grid) + 1))
    cbar.set_label('Actions')
    cbar.set_ticks(np.arange(np.max(action_grid) + 1) + 0.5)
    cbar.set_ticklabels([f"Action {i}" for i in range(np.max(action_grid) + 1)])

    plt.savefig("most_rewarded_action_heatmap.png")

def plot_action_reward_subplots(rewards, n_acceleration_bins, n_vel_bins, n_actions):
    """
    Creates a 2x3 grid of subplots showing the reward for each action across the velocity-direction grid.
    
    :param rewards: Reward matrix of shape (n_states, n_actions).
    :param n_direction_bins: The number of direction bins.
    :param n_vel_bins: The number of velocity bins.
    :param n_actions: The number of actions to visualize.
    """
    n_states = n_acceleration_bins * n_vel_bins

    # Set up the figure and the 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Iterate over each action index to create a subplot
    for action_index in range(n_actions):
        # Initialize a grid to store the reward for the specified action per (direction, velocity) pair
        reward_grid = np.zeros((n_vel_bins, n_acceleration_bins))

        # Populate the reward grid based on the reward for the specified action
        for state_index in range(n_states):
            velocity_bin = state_index // n_acceleration_bins
            direction_bin = state_index % n_acceleration_bins

            # Extract the reward for the specified action
            reward_grid[velocity_bin, direction_bin] = rewards[state_index, action_index]

        # Plotting the heatmap using imshow in the appropriate subplot
        ax = axes[action_index]
        img = ax.imshow(reward_grid, cmap='viridis_r', aspect='auto')
        ax.set_title(f"Action {action_index}")
        ax.set_xticks(ticks=np.arange(0, n_acceleration_bins)[::3], labels=np.arange(-3000, 2250, step=250)[::3], fontsize=12)
        ax.set_yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
        ax.set_xlabel("Acceleration Bins", fontsize=14)
        ax.set_ylabel("Velocity Bins", fontsize=14)

    # Add a color bar to the last subplot, shared across all subplots
    fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig("action_reward_subplots.png")

def plot_velocity_action_reward_heatmap(rewards, n_acceleration_bins, n_vel_bins):
    """
    Creates a heatmap showing the reward distribution across velocity bins and actions.
    
    :param rewards: Reward matrix of shape (n_states, n_actions).
    :param n_direction_bins: The number of direction bins.
    :param n_vel_bins: The number of velocity bins.
    """
    n_states = n_acceleration_bins * n_vel_bins
    n_actions = rewards.shape[1]

    # Initialize a grid to store the aggregated reward per (velocity bin, action)
    reward_grid = np.zeros((n_vel_bins, n_actions))

    # Populate the reward grid by aggregating over direction bins
    for state_index in range(n_states):
        velocity_bin = state_index // n_acceleration_bins
        # Sum rewards across direction bins for each velocity bin and action
        reward_grid[velocity_bin, :] += rewards[state_index, :]

    # Normalize by the number of direction bins to get an average if needed
    reward_grid /= n_acceleration_bins

    # Plotting the heatmap using imshow
    plt.figure(figsize=(10, 8))
    plt.imshow(reward_grid, cmap='viridis_r', aspect='auto')
    plt.title("Reward Heatmap: Velocity vs. Action", fontsize=16)
    plt.yticks(ticks=np.arange(0, n_vel_bins)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
    plt.xlabel("Actions", fontsize=14)
    plt.ylabel("Velocity Bins", fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig("velocity_action_reward_heatmap.png")

def plot_acceleration_action_reward_heatmap(rewards, n_acceleration_bins, n_vel_bins):
    """
    Creates a heatmap showing the reward distribution across direction bins and actions.
    
    :param rewards: Reward matrix of shape (n_states, n_actions).
    :param n_direction_bins: The number of direction bins.
    :param n_vel_bins: The number of velocity bins.
    """
    n_states = n_acceleration_bins * n_vel_bins
    n_actions = rewards.shape[1]

    # Initialize a grid to store the aggregated reward per (direction bin, action)
    reward_grid = np.zeros((n_acceleration_bins, n_actions))

    # Populate the reward grid by aggregating over velocity bins
    for state_index in range(n_states):
        direction_bin = state_index % n_acceleration_bins
        # Sum rewards across velocity bins for each direction bin and action
        reward_grid[direction_bin, :] += rewards[state_index, :]

    # Normalize by the number of velocity bins to get an average if needed
    reward_grid /= n_vel_bins

    # Plotting the heatmap using imshow
    plt.figure(figsize=(10, 8))
    plt.imshow(reward_grid, cmap='viridis_r', aspect='auto')
    plt.title("Reward Heatmap: Acceleration vs. Action", fontsize=16)
    plt.yticks(ticks=np.arange(0, n_acceleration_bins)[::3], labels=np.arange(-3000, 2250, step=250)[::3], minor=False, fontsize=12)
    plt.xlabel("Actions", fontsize=14)
    plt.ylabel("Acceleration Bins", fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig("acceleration_action_reward_heatmap.png")