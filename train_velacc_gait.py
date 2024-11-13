import numpy as np
import pandas as pd
import maxent
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

# Load the configuration file
with open('configs/irl.yml') as file:
    v = yaml.load(file, Loader=yaml.FullLoader)

# check if there is test_folder, if not create one
test_folder = v['test_folder']
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

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
n_acceleration_bins = data['Acceleration Bin'].nunique()
# Medauroidea has no bin5 for acceleration, Carausius has no bin21 for acceleration
if source in ['CarausiusC00', 'MedauroideaC00', 'C00', 'C00T']:
    n_acceleration_bins = n_acceleration_bins + 1
n_gait_categories = data['Gait Category'].nunique()
print("---------------------------------")
print(f"Velocity bins: {n_velocity_bins}")
print(f"Acceleration bins: {n_acceleration_bins}")
print(f"Gait categories: {n_gait_categories}")
print("---------------------------------")

# Create a feature matrix (n_states, n_dimensions)
n_states = n_velocity_bins * n_acceleration_bins
d_states = n_velocity_bins + n_acceleration_bins
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
    state_index = int((row['Velocity Bin']-1) * n_acceleration_bins + (row['Acceleration Bin']-1))
    # set the one-hot encoding (column index)
    feature_matrix[state_index, (row['Acceleration Bin']-1)] = 1
    feature_matrix[state_index, n_velocity_bins + (row['Acceleration Bin']-1)] = 1

def generate_trajectory(data, n_acceleration_bins):
    trajectory = []
    for index, row in data.iterrows():
        state_index = int((row['Velocity Bin'] - 1) * n_acceleration_bins + (row['Acceleration Bin'] - 1))
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

# Generate trajectories from the dataset
'''flatten_traj'''
trajectories = []
for index, row in data.iterrows():
    state_index = int((row['Velocity Bin']-1) * n_acceleration_bins + (row['Acceleration Bin']-1))
    action = int(row['Gait Category'])
    trajectories.append([(state_index, action)])
trajectories = np.array(trajectories)
# reshape the trajectories to (1, len_trajectories, 2)
len_trajectories = trajectories.shape[0]
trajectories = trajectories.reshape(1, len_trajectories, 2)
# # trajectories = trajectories.tolist()
# # print("Trajectories: ", len(trajectories), len(trajectories[0]), len(trajectories[0][0]))

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
n_bin1 = n_acceleration_bins
n_bin2 = n_velocity_bins
n_bins = [n_bin1, n_bin2]
lable_bin1 = "Acceleration Bins"
lable_bin2 = "Velocity Bins"
labels = [lable_bin1, lable_bin2]

plot_transition_heatmaps(transition_probabilities, test_folder)


# train irl
feature_matrix = torch.tensor(feature_matrix, device=device, dtype=torch.float32).to(device)
transition_probabilities = torch.tensor(transition_probabilities, device=device, dtype=torch.float32).to(device)
rewards = maxentirl_gpu(feature_matrix, n_actions, discount, transition_probabilities, 
                                        trajectories, epochs, learning_rate, n_bins, labels, test_folder, device)
#Output the inferred rewards
print("Inferred Rewards:", rewards.shape)
# Save the inferred rewards as a CSV file
np.savetxt(test_folder+'inferred_rewards.csv', rewards, delimiter=',')


# # evaluate the policy
# rewards = np.loadtxt(test_folder+'inferred_rewards.csv', delimiter=',')
# q_values = maxent.find_policy(n_states, rewards, n_actions, discount, transition_probabilities)
# print("Q-values shape: ", q_values.shape)
# # save the q_values as a CSV file
# np.savetxt(test_folder+'q_values_maxent_velocity.csv', q_values, delimiter=',')
# plot_most_rewarded_action(q_values, n_bin1, n_bin2, lable_bin1, lable_bin2, test_folder)
# plot_q_table(q_values, test_folder)
# plot_action_reward_subplots(q_values, n_bin1, n_bin2, n_actions, lable_bin1, lable_bin2, test_folder)
# plot_singlestate_action(q_values, n_states, n_bin1, lable_bin1, test_folder)
# plot_singlestate_action(q_values, n_states, n_bin2, lable_bin2, test_folder)