import numpy as np
import pandas as pd
from gridworld import CustomMDP as MDP
from maxent import maxentirl
import matplotlib.pyplot as plt
import seaborn as sns
from plot_evaluate import *
import torch
import os

# Load the dataset
# ['CarausiusC00', 'AretaonC00', 'MedauroideaC00', 'MedauroideaC00T', 'C00', 'C00T']
source = 'CarausiusC00'
data = pd.read_csv('expert_demonstration/expert/'+source+'.csv')

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
test_folder = 'test_folder/maxent/test/'
n_bin1=n_direction_bins
n_bin2=n_velocity_bins
n_bins = [n_bin1, n_bin2]
label_bin1="Direction Bins"
label_bin2="Velocity Bins"
labels = [label_bin1, label_bin2]


# # check if there is test_folder, if not create one
# if not os.path.exists(test_folder):
#     os.makedirs(test_folder)
# plot_transition_heatmaps(transition_probabilities, test_folder)

# # train irl
# rewards = maxentirl(feature_matrix, mdp.n_actions, discount, transition_probabilities, 
#                                         trajectories, epochs, learning_rate, n_bins, labels, test_folder)
# #Output the inferred rewards
# print("Inferred Rewards:", rewards.shape)
# # Save the inferred rewards as a CSV file
# np.savetxt(test_folder+'inferred_rewards_maxent_direction.csv', rewards, delimiter=',')


# evaluate the policy
rewards = np.loadtxt(test_folder+'inferred_rewards_maxent_direction.csv', delimiter=',')
q_values = maxent.find_policy(n_states, rewards, n_actions, discount, transition_probabilities)
print("Q-values shape: ", q_values.shape)
# save the q_values as a CSV file
np.savetxt(test_folder+'q_values_maxent_direction.csv', q_values, delimiter=',')
plot_most_rewarded_action(q_values, n_bin1, n_bin2, lable_bin1, lable_bin2, test_folder)
plot_q_table(q_values, test_folder)
plot_action_reward_subplots(q_values, n_bin1, n_bin2, n_actions, lable_bin1, lable_bin2, test_folder)
plot_singlestate_action(q_values, n_states, n_bin1, lable_bin1, test_folder)
plot_singlestate_action(q_values, n_states, n_bin2, lable_bin2, test_folder)