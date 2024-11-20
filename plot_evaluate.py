import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



'''Logging to a file'''
# Redirect stdout to the log file
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Only log non-empty messages
            self.level(message)

    def flush(self):
        pass  # Required for compatibility



'''Plotting functions for the 2-dimension states'''
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

def plot_most_rewarded_action(q_values, n_bin1, n_bin2, label_bin1, label_bin2, test_folder):
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
    plt.xlabel(label_bin1)
    plt.ylabel(label_bin2)
    plt.savefig(test_folder+'most_rewarded_action_heatmap.png')

def plot_action_reward_subplots(q_values, n_bin1, n_bin2, n_actions, label_bin1, label_bin2, test_folder):
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
        # ax = axes[action_index]
        ax = axes[action_index+1 if n_actions==5 else action_index]  # Shift the index for 5 actions        
        img = ax.imshow(reward_grid, cmap='viridis', aspect='auto')
        # ax.set_title(f"Action {action_index}", fontsize=16)
        ax.set_title(f"Action {action_index+1 if n_actions==5 else action_index}", fontsize=16)
        # ax.set_xticks(ticks=np.arange(0, n_bin1), labels=np.arange(-20, 5, step=5), fontsize=12)
        # ax.set_yticks(ticks=np.arange(0, n_bin2)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
        ax.set_xlabel(label_bin1, fontsize=14)
        ax.set_ylabel(label_bin2, fontsize=14)
    # Add a color bar to the last subplot, shared across all subplots
    # change the ax position
    # cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # Leave the first subplot empty if there are only 5 actions
    if n_actions == 5:
        axes[0].axis('off')  # Hide the first subplot (action 0)
    fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(test_folder+"action_reward_subplots.png")

def plot_singlestate_action(q_values, n_states, n_bin, label_bin, test_folder):
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
    plt.title("Reward Heatmap: "+label_bin+" vs. Action", fontsize=16)
    # plt.yticks(ticks=np.arange(0, n_bin), labels=np.arange(-20, 5, step=5), fontsize=12)
    plt.xlabel("Actions", fontsize=14)
    plt.ylabel(label_bin, fontsize=14)
    if q_values.shape[1] == 5:
        plt.xticks(ticks=np.arange(q_values.shape[1]), labels=np.arange(1, q_values.shape[1]+1))
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+"Action "+label_bin+"Reward Heatmap.png")



'''Plotting functions for the 4-dimension states'''
def plot_most_rewarded_action4d(q_values, n_bin1, n_bin2, n_bin3, n_bin4, label_1, label_2, test_folder):
    # Find the action with the highest Q-value for each state
    most_rewarded_action = np.argmax(q_values, axis=1)
    print("Most rewarded action shape: ", most_rewarded_action.shape)
    # Adjust the annotation for 5 actions case to be 1 to 5
    if  q_values.shape[1] == 5:
        most_rewarded_action = most_rewarded_action + 1
    # Reshape the 4D state space to 2D by fixing two dimensions
    most_rewarded_action_4d = most_rewarded_action.reshape(n_bin4, n_bin3, n_bin2, n_bin1)
    if label_1 == "HS Left Bins" and label_2 == "HS Right Bins":
        most_rewarded_action_2d = most_rewarded_action_4d[0, 0, :, :]
        # Plot the heatmap (reshaping if the states are grid-like, otherwise just plot)
        plt.figure(figsize=(10, 8))
        sns.heatmap(most_rewarded_action_2d, cmap="YlGnBu", annot=True)
        plt.title("Most Rewarded Action for Each State")
        plt.xlabel(label_1)
        plt.ylabel(label_2)
        plt.savefig(test_folder+'most_rewarded_action_heatmap_HS.png')
    elif label_1 == "SP Left Bins" and label_2 == "SP Right Bins":
        most_rewarded_action_2d = most_rewarded_action_4d[:, :, 0, 0]
        # Plot the heatmap (reshaping if the states are grid-like, otherwise just plot)
        plt.figure(figsize=(10, 8))
        sns.heatmap(most_rewarded_action_2d, cmap="YlGnBu", annot=True)
        plt.title("Most Rewarded Action for Each State")
        plt.xlabel(label_1)
        plt.ylabel(label_2)
        plt.savefig(test_folder+'most_rewarded_action_heatmap_SP.png')

def plot_most_rewarded_action_4d_subplots(q_values, n_bin1, n_bin2, n_bin3, n_bin4, label_bin1, label_bin2, test_folder):
    # Find the action with the highest Q-value for each state
    most_rewarded_action = np.argmax(q_values, axis=1)
    print("Most rewarded action shape: ", most_rewarded_action.shape)
    # Adjust the annotation for 5 actions case to be 1 to 5
    if q_values.shape[1] == 5:
        most_rewarded_action = most_rewarded_action + 1
    # Reshape most_rewarded_action array to its original 4D shape
    most_rewarded_action_4d = most_rewarded_action.reshape(n_bin4, n_bin3, n_bin2, n_bin1)
    # Create subplots for different slices of the 4D state space
    fig, axes = plt.subplots(n_bin3, n_bin4, figsize=(15, 15))
    fig.suptitle("Most Rewarded Action for Each State for Different Fixed Values", fontsize=16)
    for i in range(n_bin3):
        for j in range(n_bin4):
            ax = axes[i, j]
            sns.heatmap(most_rewarded_action_4d[j, i, :, :], cmap="YlGnBu", annot=True, ax=ax)
            ax.set_title(f"Fixed Indices: n_bin3={i}, n_bin4={j}")
            ax.set_xlabel(label_bin1)
            ax.set_ylabel(label_bin2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(test_folder + 'most_rewarded_action_heatmap_subplots.png')

def plot_action_reward_subplots4d(q_values, n_bin1, n_bin2, n_bin3, n_bin4, n_actions, label_1, label_2, test_folder):
    # Set up the figure and the 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    # Define fixed values for bin3 and bin4 to create a slice for visualization
    fixed_bin = 0  # Example fixed value for bin3
    # Iterate over each action index to create a subplot
    if label_1 == "HS Left Bins" and label_2 == "HS Right Bins":
        for action_index in range(n_actions):
            # Initialize a grid to store the reward for the specified action per (direction, velocity) pair
            reward_grid = np.zeros((n_bin2, n_bin1))
            # Populate the reward grid based on the reward for the specified action
            for bin2 in range(n_bin2):
                for bin1 in range(n_bin1):
                    # Calculate the state index based on the bin locations of all 4 dimensions
                    state_index = (fixed_bin * n_bin3 * n_bin2 * n_bin1) + (fixed_bin * n_bin2 * n_bin1) + (bin2 * n_bin1) + bin1
                    reward_grid[bin2, bin1] = q_values[state_index, action_index]
            # Plotting the heatmap using imshow in the appropriate subplot
            # ax = axes[action_index]
            ax = axes[action_index+1 if n_actions==5 else action_index]  # Shift the index for 5 actions        
            img = ax.imshow(reward_grid, cmap='viridis', aspect='auto')
            # ax.set_title(f"Action {action_index}", fontsize=16)
            ax.set_title(f"Action {action_index+1 if n_actions==5 else action_index}", fontsize=16)
            # ax.set_xticks(ticks=np.arange(0, n_bin1), labels=np.arange(-20, 5, step=5), fontsize=12)
            # ax.set_yticks(ticks=np.arange(0, n_bin2)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
            ax.set_xlabel(label_1, fontsize=14)
            ax.set_ylabel(label_2, fontsize=14)
        # Add a color bar to the last subplot, shared across all subplots
        # change the ax position
        # cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # Leave the first subplot empty if there are only 5 actions
        if n_actions == 5:
            axes[0].axis('off')  # Hide the first subplot (action 0)
        fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig(test_folder+"action_reward_subplots_HS.png")
    elif label_1 == "SP Left Bins" and label_2 == "SP Right Bins":
        for action_index in range(n_actions):
            # Initialize a grid to store the reward for the specified action per (direction, velocity) pair
            reward_grid = np.zeros((n_bin4, n_bin3))
            # Populate the reward grid based on the reward for the specified action
            for bin4 in range(n_bin4):
                for bin3 in range(n_bin3):
                    # Calculate the state index based on the bin locations of all 4 dimensions
                    state_index = (bin4 * n_bin3 * n_bin2 * n_bin1) + (bin3 * n_bin2 * n_bin1) + (fixed_bin * n_bin1) + fixed_bin
                    reward_grid[bin4, bin3] = q_values[state_index, action_index]
            # Plotting the heatmap using imshow in the appropriate subplot
            # ax = axes[action_index]
            ax = axes[action_index+1 if n_actions==5 else action_index]  # Shift the index for 5 actions        
            img = ax.imshow(reward_grid, cmap='viridis', aspect='auto')
            # ax.set_title(f"Action {action_index}", fontsize=16)
            ax.set_title(f"Action {action_index+1 if n_actions==5 else action_index}", fontsize=16)
            # ax.set_xticks(ticks=np.arange(0, n_bin1), labels=np.arange(-20, 5, step=5), fontsize=12)
            # ax.set_yticks(ticks=np.arange(0, n_bin2)[::3], labels=np.arange(0, 140, step=5)[::3], fontsize=12)
            ax.set_xlabel(label_1, fontsize=14)
            ax.set_ylabel(label_2, fontsize=14)
        # Add a color bar to the last subplot, shared across all subplots
        # change the ax position
        # cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # Leave the first subplot empty if there are only 5 actions
        if n_actions == 5:
            axes[0].axis('off')  # Hide the first subplot (action 0)
        fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig(test_folder+"action_reward_subplots_SP.png")       