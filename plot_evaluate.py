import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_most_rewarded_action(q_values, n_bin1, n_bin2, lable_bin1, lable_bin2, test_folder):
    # Find the action with the highest Q-value for each state
    most_rewarded_action = np.argmax(q_values, axis=1)
    print("Most rewarded action shape: ", most_rewarded_action.shape)
    # Adjust the annotation for 5 actions case to be 1 to 5
    if  q_values.shape[1] == 5:
        most_rewarded_action = most_rewarded_action + 1
    # Plot the heatmap (reshaping if the states are grid-like, otherwise just plot)
    plt.figure(figsize=(10, 8))
    sns.heatmap(most_rewarded_action.reshape(n_bin2, n_bin1), cmap="YlGnBu", annot=True)
    plt.title("Most Rewarded Action for Each State")
    plt.xlabel(lable_bin1)
    plt.ylabel(lable_bin2)
    plt.savefig(test_folder+'most_rewarded_action_heatmap.png')

def plot_q_table(q_values, test_folder):
    plt.figure(figsize=(10, 8))
    plt.title("Q-Table Heatmap (State-Action Rewards)", fontsize=16)
    if q_values.shape[1] == 5:
        plt.xticks(ticks=np.arange(q_values.shape[1]), labels=np.arange(1, q_values.shape[1]+1))
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
    if q_values.shape[1] == 5:
        plt.xticks(ticks=np.arange(q_values.shape[1]), labels=np.arange(1, q_values.shape[1]+1))
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+"Action "+lable_bin+"Reward Heatmap.png")