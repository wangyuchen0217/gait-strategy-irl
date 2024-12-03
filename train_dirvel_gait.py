import numpy as np
import pandas as pd
import maxent_gpu
from maxent_gpu import maxentirl as maxentirl_gpu
import matplotlib.pyplot as plt
import seaborn as sns
from plot_evaluate import *
import torch
import os
import sys
import yaml
import logging
from datetime import datetime
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance

# Load the configuration file
with open('configs/irl.yml') as file:
    v = yaml.load(file, Loader=yaml.FullLoader)
mode = v['mode']

# check if there is test_folder, if not create one
test_folder = v['test_folder']
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

if mode == 'train':
    # Set up logging configuration
    current_time = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
    log_filename = f"{test_folder}training_process.log"
    logging.basicConfig(
        filename=log_filename,    
        level=logging.INFO,
        format='%(message)s',
        filemode='w'
        )
    sys.stdout = LoggerWriter(logging.info)
    print(f"Logging started at {current_time}")

    # Set the device
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(v['cuda']))
    else:
        print("Running on CPU")
    print(f"Process ID: {os.getpid()}")

# Load the dataset
source = v['data_source']
data = pd.read_csv('expert_demonstration/expert/'+source+'.csv')

# Prepare the MDP
n_velocity_bins = data['Velocity Bin'].nunique()
n_direction_bins = data['Direction Bin'].nunique()
n_gait_categories = data['Gait Category'].nunique()
print("---------------------------------")
print(f"Velocity bins: {n_velocity_bins}")
print(f"Direction bins: {n_direction_bins}")
print(f"Gait categories: {n_gait_categories}")
print("---------------------------------")

# Create a feature matrix (n_states, n_dimensions)
n_states = n_velocity_bins * n_direction_bins
d_states = n_velocity_bins + n_direction_bins
n_actions = n_gait_categories
feature_matrix = np.zeros((n_states, d_states))
print(f"Number of states: {n_states}")
print(f"Number of actions: {n_actions}")
print(f"Dimension of states: {d_states}")
print(f"Rewards shape: ({n_states},)")
print(f"Feature matrix shape: {feature_matrix.shape}")

# Populate the feature matrix (one-hot encoding)
for index, row in data.iterrows():
    # set the row index
    state_index = int((row['Velocity Bin']-1) * n_direction_bins + (row['Direction Bin']-1))
    # set the one-hot encoding (column index)
    feature_matrix[state_index, (row['Direction Bin']-1)] = 1
    feature_matrix[state_index, n_velocity_bins + (row['Direction Bin']-1)] = 1

def generate_trajectory(data, n_direction_bins):
    trajectories = []
    for index, row in data.iterrows():
        state_index = int((row['Velocity Bin']-1) * n_direction_bins + (row['Direction Bin']-1))
        action = int(row['Gait Category'])
        trajectories.append([(state_index, action)])
    trajectories = np.array(trajectories)
    # reshape the trajectories to (1, len_trajectories, 2)
    len_trajectories = trajectories.shape[0]
    trajectories = trajectories.reshape(1, len_trajectories, 2)
    return trajectories

def build_transition_matrix_from_indices(data, n_states, n_actions):
    transition_counts = np.zeros((n_states, n_actions, n_states))
    # Iterate over the data and infer the next state from the next tuple
    for i in range(len(data) - 1):
        state, action = data[i]          # Current state and action
        next_state, _ = data[i + 1]      # Infer the next state from the next tuple
        # Increment the count for this transition
        # Medauroidea only has 5 actions (1-5)
        if n_actions < 6:
            action -= 1
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

# Generate trajectories from the dataset: flatten_traj
trajectories = generate_trajectory(data, n_direction_bins)

transition_probabilities = build_transition_matrix_from_indices(trajectories[0], n_states, n_actions)
print(f"Transition probabilities shape: {transition_probabilities.shape}")
print("---------------------------------")

def plot_transition_heatmaps(transition_probabilities, test_folder):
    plt.figure(figsize=(18, 12))
    if n_actions == 6:
        for action in range(6):
            plt.subplot(2, 3, action+1)
            sns.heatmap(transition_probabilities[:, action, :], cmap="YlGnBu", annot=False)
            plt.title(f"Transition Probabilities for Action {action+1}")
            plt.xlabel("Next State Index")
            plt.ylabel("Current State Index")
    else:
        for action in range(5):
            plt.subplot(2, 3, action+1)
            sns.heatmap(transition_probabilities[:, action, :], cmap="YlGnBu", annot=False)
            plt.title(f"Transition Probabilities for Action {action+1}")
            plt.xlabel("Next State Index")
            plt.ylabel("Current State Index")
    plt.tight_layout()
    plt.savefig(test_folder+'transition_heatmaps.png')


# Apply MaxEnt IRL
epochs = 1000
learning_rate = 0.01
discount = 0.9
n_bin1=n_direction_bins
n_bin2=n_velocity_bins
n_bins = [n_bin1, n_bin2]
label_bin1="Direction Bins"
label_bin2="Velocity Bins"
labels = [label_bin1, label_bin2]

plot_transition_heatmaps(transition_probabilities, test_folder)

if mode == 'train':
    # train irl
    feature_matrix = torch.tensor(feature_matrix, device=device, dtype=torch.float32).to(device)
    transition_probabilities = torch.tensor(transition_probabilities, device=device, dtype=torch.float32).to(device)
    trajectories = torch.tensor(trajectories, device=device, dtype=torch.int64).to(device)
    rewards = maxentirl_gpu(feature_matrix, n_actions, discount, transition_probabilities, 
                                            trajectories, epochs, learning_rate, n_bins, labels, test_folder, device)
    #Output the inferred rewards
    print("Inferred Rewards:", rewards.shape)
    # Save the inferred rewards as a pt file
    torch.save(rewards, test_folder+'inferred_rewards.pt')


if mode == 'evaluate':
    # evaluate the policy
    rewards = torch.load(test_folder+'inferred_rewards.pt', map_location=f"cuda:{v['cuda']}")
    rewards = rewards.cpu().clone().numpy()
    q_values = maxent_gpu.find_policy(n_states, rewards, n_actions, discount, transition_probabilities)
    print("Q-values shape: ", q_values.shape)
    # save the q_values as a CSV file
    np.savetxt(test_folder+'q_values_maxent_direction.csv', q_values, delimiter=',')
    plot_most_rewarded_action(q_values, n_bin1, n_bin2, label_bin1, label_bin2, test_folder)
    plot_q_table(q_values, test_folder)
    plot_action_reward_subplots(q_values, n_bin1, n_bin2, n_actions, label_bin1, label_bin2, test_folder)
    plot_singlestate_action(q_values, n_states, n_bin1, label_bin1, test_folder)
    plot_singlestate_action(q_values, n_states, n_bin2, label_bin2, test_folder)


def evaluate_trajectory_metrics(expert_trajectory, replicated_trajectory):
    # Ensure both trajectories are of the same length
    assert len(expert_trajectory) == len(replicated_trajectory), "Trajectories must be of equal length for comparison."
    expert_trajectory = expert_trajectory.reshape(-1, 1)
    replicated_trajectory = replicated_trajectory.reshape(-1, 1)
    # Modified Hausdorff Distance (MHD)
    def modified_hausdorff_distance(a, b):
        forward_hausdorff = directed_hausdorff(a, b)[0]
        backward_hausdorff = directed_hausdorff(b, a)[0]
        return max(forward_hausdorff, backward_hausdorff)
    mhd = modified_hausdorff_distance(expert_trajectory, replicated_trajectory)
    # Root Mean Square Percentage Error (RMSPE)
    rmspe = np.sqrt(mean_squared_error(expert_trajectory, replicated_trajectory)) / np.mean(expert_trajectory) * 100
    # Sliced Wasserstein Distance (SWD)
    swd = wasserstein_distance(expert_trajectory.flatten(), replicated_trajectory.flatten())
    print(f"Modified Hausdorff Distance (MHD): {mhd}")
    print(f"Root Mean Square Percentage Error (RMSPE): {rmspe}%")
    print(f"Sliced Wasserstein Distance (SWD): {swd}")
    return mhd, rmspe, swd

if mode == 'test':
    # Use the expert trajectory as the ground truth
    expert_trajectory = trajectories
    state_indices = expert_trajectory[0, :, 0]
    actions = expert_trajectory[0, :, 1]
    np.savetxt(test_folder+'trajectories.csv', expert_trajectory[0], delimiter=',')
    # load the q_values
    q_values = np.loadtxt(test_folder+'q_values_maxent_direction.csv', delimiter=',')
    # Generate the replicated trajectory
    # # Basic one
    # replicated_trajectory = []
    # action_posibility = []
    # for i in range(len(state_indices)):
    #     action_probabilities = q_values[state_indices[i]]
    #     action_posibility.append(action_probabilities)
    #     # Select the action with the highest probability (greedy policy)
    #     action = np.argmax(action_probabilities)
    #     replicated_trajectory.append(action)
    # np.savetxt(test_folder+'replicated_trajectory.csv', np.array(replicated_trajectory), delimiter=',')
    # np.savetxt(test_folder+'action_posibility.csv', np.array(action_posibility), delimiter=',')

    # Policy refinement
    replicated_trajectory = []
    action_probability = []
    max_counts = []
    for i in range(len(state_indices)):
        action_probabilities = q_values[state_indices[i]]
        action_probability.append(action_probabilities)
        # check the number of actions with the highest probability
        max_value = np.max(action_probabilities)
        max_count = np.sum(action_probabilities == max_value)
        max_counts.append(max_count)
        # Select the action with the highest probability (greedy policy)
        if max_count > 1:
            # action = np.random.choice(np.where(action_probabilities == max_value)[0])
            action = previous_action if previous_action is not None else np.argmax(action_probabilities)
            if action not in np.where(action_probabilities == max_value)[0]:
                print("Action not in the max values:", i, action)
        else:
            action = np.argmax(action_probabilities)
        previous_action = action
        replicated_trajectory.append(action)
    np.savetxt(test_folder+'action_probability.csv', action_probability, delimiter=',')
    np.savetxt(test_folder+'max_counts.csv', max_counts, delimiter=',')

    # Convert to numpy array
    replicated_trajectory = np.array(replicated_trajectory)
    print("Expert Trajectory Shape: ", actions.shape)
    print("Replicated Trajectory Shape: ", replicated_trajectory.shape)

    # Plot a heat map to show the trajectory using imshow
    plt.figure(figsize=(10, 3))
    plt.imshow(q_values[state_indices].T, cmap="plasma", aspect='auto')
    plt.title("Heatmap of Action Probabilities along the Expert Trajectory")
    plt.xlabel("Trajectory Step Index")
    plt.ylabel("Action Index")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(test_folder+'actions_probability_trajectory.png')

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.eventplot([np.where(replicated_trajectory == i)[0] for i in range(6)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
    plt.yticks(range(6), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4", "Action 5"])
    plt.xlabel("Trajectory Step Index")
    plt.ylabel("Action")
    plt.title("Actions along the Replicated Trajectory")
    plt.subplot(2, 1, 2)
    if n_actions == 6:
        plt.eventplot([np.where(actions == i)[0] for i in range(6)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
        plt.yticks(range(6), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4", "Action 5"])
    elif n_actions == 5:
        plt.eventplot([np.where(actions == i)[0] for i in range(5)], lineoffsets=1, linelengths=0.5, colors=['red', 'blue', 'green', 'orange', 'purple'])
        plt.yticks(range(5), labels=["Action 0", "Action 1", "Action 2", "Action 3", "Action 4"])
    plt.xlabel("Trajectory Step Index")
    plt.ylabel("Action")
    plt.title("Actions along the Expert Trajectory")
    plt.tight_layout()
    plt.savefig(test_folder+'actions_along_trajectories.png')

    mhd, rmspe, swd = evaluate_trajectory_metrics(actions, replicated_trajectory)
