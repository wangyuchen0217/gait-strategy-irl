import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from scipy import stats
from scipy.stats import wasserstein_distance



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
    ax = sns.heatmap(most_rewarded_action.reshape(n_bin2, n_bin1), cmap="YlGnBu", annot=True)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    plt.title("Most Rewarded Action for Each State", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(label_bin1, fontsize=18)
    plt.ylabel(label_bin2, fontsize=18)
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
        ax.set_title(f"Action {action_index+1 if n_actions==5 else action_index}", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel(label_bin1, fontsize=18)
        ax.set_ylabel(label_bin2, fontsize=18)
    # Add a color bar to the last subplot, shared across all subplots
    # change the ax position
    # cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # Leave the first subplot empty if there are only 5 actions
    if n_actions == 5:
        axes[0].axis('off')  # Hide the first subplot (action 0)
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
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
    plt.savefig(test_folder+"Action "+label_bin+" Reward Heatmap.png")



'''Plotting functions for the 4-dimension states'''
def plot_most_rewarded_action_4d_subplots(q_values, n_bin1, n_bin2, n_bin3, n_bin4, 
                                          label_bin1, label_bin2, label_bin3, label_bin4, test_folder):
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
    fig.suptitle("Most Rewarded Action for Each State", fontsize=18)
    for i in range(n_bin3):
        for j in range(n_bin4):
            ax = axes[i, j]
            sns.heatmap(most_rewarded_action_4d[j, i, :, :], cmap="YlGnBu", annot=True, ax=ax)
            ax.set_title(f"{label_bin3}:{i}, {label_bin4}:{j}", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlabel(label_bin1, fontsize=12)
            ax.set_ylabel(label_bin2, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(test_folder + 'most_rewarded_action_heatmap_subplots.png')

def plot_action_reward_all_combinations(q_values, n_bin1, n_bin2, n_bin3, n_bin4, n_actions, 
                                        label_bin1, label_bin2, label_bin3, label_bin4, test_folder):
    # Reshape q_values into its 4D form for easier slicing
    q_values_4d = q_values.reshape((n_bin4, n_bin3, n_bin2, n_bin1, n_actions))
    # Set up the figure for subplots for all combinations of bin3 and bin4
    fig, axes = plt.subplots(n_bin3, n_bin4, figsize=(18, 18))
    fig.suptitle("Max Reward for All Actions in Each State Combination", fontsize=16)
    axes = axes.flatten()
    # Iterate over each combination of `bin3` and `bin4` to create a subplot for each
    for i in range(n_bin3):
        for j in range(n_bin4):
            ax = axes[j + i * n_bin4]  # Calculate the proper index for the subplot
            # Initialize a grid to store the reward for the specified action per (bin1, bin2) pair
            reward_grid = np.zeros((n_bin2, n_bin1))
            # Populate the reward grid for the current `bin3` and `bin4` combination
            for bin2 in range(n_bin2):
                for bin1 in range(n_bin1):
                    reward_grid[bin2, bin1] = np.max(q_values_4d[j, i, bin2, bin1, :])  # Find the max reward across actions for each state
            # Plotting the heatmap for the current slice
            sns.heatmap(reward_grid, cmap="viridis", ax=ax)
            ax.set_title(f"{label_bin3}:{i}, {label_bin4}:{j}")
            ax.set_xlabel(label_bin1)
            ax.set_ylabel(label_bin2)
    # Adjust the layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(test_folder + "action_reward_all_combinations.png")

def plot_action_reward_individual(q_values, n_bin1, n_bin2, n_bin3, n_bin4, n_actions, 
                                  label_bin1, label_bin2, label_bin3, label_bin4, test_folder):
    # Reshape q_values into its 4D form for easier slicing
    q_values_4d = q_values.reshape((n_bin4, n_bin3, n_bin2, n_bin1, n_actions))
    # Iterate over each action to create a separate figure for each action
    for action_index in range(n_actions):
        # Set up the figure for subplots for all combinations of bin3 and bin4
        fig, axes = plt.subplots(n_bin3, n_bin4, figsize=(18, 18))
        fig.suptitle(f"Reward for Action {action_index} in Each State Combination", fontsize=16)
        axes = axes.flatten()
        # Iterate over each combination of `bin3` and `bin4` to create a subplot for each
        for i in range(n_bin3):
            for j in range(n_bin4):
                ax = axes[j + i * n_bin4]  # Calculate the proper index for the subplot
                # Initialize a grid to store the reward for the specified action per (bin1, bin2) pair
                reward_grid = np.zeros((n_bin2, n_bin1))
                # Populate the reward grid for the current `bin3` and `bin4` combination
                for bin2 in range(n_bin2):
                    for bin1 in range(n_bin1):
                        reward_grid[bin2, bin1] = q_values_4d[j, i, bin2, bin1, action_index]  # Extract reward for the specific action
                # Plotting the heatmap for the current slice
                sns.heatmap(reward_grid, cmap="viridis", ax=ax)
                ax.set_title(f"{label_bin3}={i}, {label_bin4}={j}")
                ax.set_xlabel(label_bin1)
                ax.set_ylabel(label_bin2)
        # Adjust the layout and save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{test_folder}action_reward_subplots_action_{action_index}.png")
        plt.close(fig)  # Close the figure to avoid overlap in subsequent iterations


'''Plotting functions for test'''
def plot_replicated_action_prob(q_values, state_indices, test_folder):
    plt.figure(figsize=(10, 3))
    plt.imshow(q_values[state_indices].T, cmap="plasma", aspect='auto')
    # plt.title("Heatmap of Action Probabilities along the Replicated Trajectory", fontsize=16)
    plt.title("Policy vel-dir: Action Probabilities Distribution along the Trajectory", fontsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.xlabel("Trajectory Step Index", fontsize=14)
    plt.ylabel("Action Index", fontsize=14)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(test_folder+'actions_probability_trajectory.png')

def argmax_replicated_traj(replicated_trajectory, actions, n_actions, test_folder):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.eventplot([np.where(replicated_trajectory == i)[0] for i in range(6)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
    plt.yticks(range(6), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4", "Action 5"])
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.xlabel("Trajectory Step Index", fontsize=14)
    plt.ylabel("Action", fontsize=14)
    plt.title("Actions along the Replicated Trajectory", fontsize=16)
    plt.subplot(2, 1, 2)
    if n_actions == 6:
        plt.eventplot([np.where(actions == i)[0] for i in range(6)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
        plt.yticks(range(6), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4", "Action 5"])
    elif n_actions == 5:
        plt.eventplot([np.where(actions == i)[0] for i in range(5)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple'])
        plt.yticks(range(5), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4"])
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.xlabel("Trajectory Step Index", fontsize=14)
    plt.ylabel("Action", fontsize=14)
    plt.title("Actions along the Expert Trajectory", fontsize=16)
    plt.tight_layout()
    plt.savefig(test_folder+'actions_along_trajectories.png')

def evaluate_action_distribution_metrics(actions, action_probability, action_of_interest):
    # Ensure input trajectories are NumPy arrays
    actions = np.array(actions)
    action_probability = np.array(action_probability)

    # Filter the trajectories for the action of interest
    expert_trajectory = np.where(actions == action_of_interest, 1.0, 0.0)
    replicated_action_prob = action_probability[:, action_of_interest]

    # print(f"Expert Action Shape: {expert_trajectory.shape}")
    # print(expert_trajectory)
    # print(f"Replicated Action Shape: {replicated_action_prob.shape}")
    # print(replicated_action_prob)

    # Reshape for compatibility with metrics
    expert_trajectory = expert_trajectory.reshape(-1, 1)
    replicated_action_prob = replicated_action_prob.reshape(-1, 1)

    # Modified Hausdorff Distance (MHD)
    def modified_hausdorff_distance(a, b):
        forward_hausdorff = directed_hausdorff(a, b)[0]
        backward_hausdorff = directed_hausdorff(b, a)[0]
        return max(forward_hausdorff, backward_hausdorff)
    mhd = modified_hausdorff_distance(expert_trajectory, replicated_action_prob)

    # Sliced Wasserstein Distance (SWD)
    swd = wasserstein_distance(
        expert_trajectory.flatten(), replicated_action_prob.flatten()
    )

    # p-value
    _,p_value = stats.ttest_ind(replicated_action_prob,expert_trajectory) 


    # Print results
    print('------')
    print(f"Metrics for Action {action_of_interest}:")
    print(f"Modified Hausdorff Distance (MHD): {mhd}")
    print(f"Sliced Wasserstein Distance (SWD): {swd}")
    print(f"p-value: {p_value}")

    # Return results as a dictionary
    return {
        "mhd": mhd,
        "swd": swd,
        "p_value": p_value,
    }