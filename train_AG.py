'''
This code is used to train the MaxEnt IRL algorithm on the antenna dataset.
states: 4D-antenna data (HS_left, SP_left, HS_right, SP_right)
actions: Gait patterns (0-5)
'''

import numpy as np
import pandas as pd
import algorithms.maxent_gpu
from algorithms.maxent_gpu import maxentirl as maxentirl_gpu
import matplotlib.pyplot as plt
import seaborn as sns
from plot_evaluate import *
from plot_test import *
import torch
import os
import sys
import yaml
import logging
from datetime import datetime
from utils import LoggerWriter

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
n_HS_left_bins = data['HS left'].nunique()
n_HS_right_bins = data['HS right'].nunique()
n_SP_left_bins = data['SP left'].nunique()
n_SP_right_bins = data['SP right'].nunique()
n_gait_categories = data['Gait Category'].nunique()
print("---------------------------------")
print(f"HS left bins: {n_HS_left_bins}")
print(f"HS right bins: {n_HS_right_bins}")
print(f"SP left bins: {n_SP_left_bins}")
print(f"SP right bins: {n_SP_right_bins}")
print(f"Gait Categories: {n_gait_categories}")
print("---------------------------------")

# Create a feature matrix (n_states, n_dimensions)
n_states = n_HS_left_bins * n_SP_left_bins * n_HS_right_bins * n_SP_right_bins
d_states = n_HS_left_bins + n_SP_left_bins + n_HS_right_bins + n_SP_right_bins
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
    state_index = int((row['HS left']-1) * n_SP_left_bins * n_HS_right_bins * n_SP_right_bins +
                        (row['SP left']-1) * n_HS_right_bins * n_SP_right_bins +
                        (row['HS right']-1) * n_SP_right_bins +
                        (row['SP right']-1))
    # set the one-hot encoding (column index)
    feature_matrix[state_index, row['HS left']-1] = 1
    feature_matrix[state_index, n_HS_left_bins + row['SP left']-1] = 1
    feature_matrix[state_index, n_HS_left_bins + n_SP_left_bins + row['HS right']-1] = 1
    feature_matrix[state_index, n_HS_left_bins + n_SP_left_bins + n_HS_right_bins + row['SP right']-1] = 1

def generate_trajectory(data, n_HS_right_bins, n_SP_left_bins, n_SP_right_bins):
    trajectories = []
    for index, row in data.iterrows():
        state_index = int(
            (row['HS left'] - 1) * n_SP_left_bins * n_HS_right_bins * n_SP_right_bins +
            (row['SP left']-1) * n_HS_right_bins * n_SP_right_bins +
            (row['HS right']-1) * n_SP_right_bins +
            (row['SP right'] - 1)
        )
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
trajectories = generate_trajectory(data, n_HS_right_bins, n_SP_left_bins, n_SP_right_bins)

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
epochs = 100
learning_rate = 0.01
discount = 0.9
n_bin1=n_HS_left_bins
n_bin2=n_SP_left_bins
n_bin3=n_HS_right_bins
n_bin4=n_SP_right_bins
n_bins=[n_bin1, n_bin2, n_bin3, n_bin4]
label_bin1="HS Left Bins"
label_bin2="SP Left Bins"
label_bin3="HS Right Bins" 
label_bin4="SP Right Bins"
labels=[label_bin1, label_bin2, label_bin3, label_bin4]

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
    q_values = algorithms.maxent_gpu.find_policy(n_states, rewards, n_actions, discount, transition_probabilities)
    print("Q-values shape: ", q_values.shape)
    # save the q_values as a CSV file
    np.savetxt(test_folder+'q_values_maxent_antenna.csv', q_values, delimiter=',')
    # plot the most rewarded action
    plot_most_rewarded_action_4d_subplots(q_values, n_bin1, n_bin2, n_bin3, n_bin4, 
                                          label_bin1, label_bin2, label_bin3, label_bin4, test_folder)
    # plot the Q-table
    plot_q_table(q_values, test_folder)
    # plot the action-reward subplots
    # plot_action_reward_all_combinations(q_values, n_bin1, n_bin2, n_bin3, n_bin4, n_actions, 
    #                                     label_bin1, label_bin2, label_bin3, label_bin4, test_folder)
    # plot_action_reward_individual(q_values, n_bin1, n_bin2, n_bin3, n_bin4, n_actions, 
    #                               label_bin1, label_bin2, label_bin3, label_bin4, test_folder)


if mode == 'test':
    # Use the expert trajectory as the ground truth
    expert_trajectory = trajectories
    state_indices = expert_trajectory[0, :, 0]
    actions = expert_trajectory[0, :, 1]
    np.savetxt(test_folder+'trajectories.csv', expert_trajectory[0], delimiter=',')
    # load the q_values
    q_values = np.loadtxt(test_folder+'q_values_maxent_antenna.csv', delimiter=',')
    # Generate the replicated trajectory
    # Basic one
    replicated_trajectory = []
    action_probability = []
    for i in range(len(state_indices)):
        action_probabilities = q_values[state_indices[i]]
        action_probability.append(action_probabilities)
    np.savetxt(test_folder+'action_probability.csv', action_probability, delimiter=',')

    # Plot a heat map to show the trajectory using imshow
    plot_replicated_action_prob(q_values, state_indices, test_folder, label="antenna")
    plot_expert_action_traj(actions, n_actions, test_folder)

    # Evaluate the action distribution metrics
    for i in range(n_actions):
        evaluate_action_distribution_metrics(actions, action_probability, action_of_interest=i)