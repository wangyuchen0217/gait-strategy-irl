import numpy as np
import pandas as pd
from gridworld import CustomMDP as MDP
from maxent import customirl as irl
from maxent import irl as maxentirl
import maxent_gpu
import matplotlib.pyplot as plt
import seaborn as sns
from plot_evaluate import *
import torch

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

def generate_trajectory(data, n_direction_bins):
    trajectory = []
    for index, row in data.iterrows():
        state_index = int((row['Velocity Bin'] - 1) * n_direction_bins + (row['Direction Bin'] - 1))
        action = int(row['Gait Category'])
        trajectory.append([state_index, action])
    return trajectory

# Generate trajectories from the dataset
'''cut_traj array'''
# t01 = data.iloc[0:2456]
# t02 = data.iloc[2456:3960]
# t03 = data.iloc[3960:5199]
# traj_01 = generate_trajectory(t01, n_direction_bins)
# traj_02 = generate_trajectory(t02, n_direction_bins)
# traj_03 = generate_trajectory(t03, n_direction_bins)
# traj_01 = np.array(traj_01)[:1239, :]
# traj_02 = np.array(traj_02)[:1239, :]
# traj_03 = np.array(traj_03)[:1239, :]
# trajectories = np.array([traj_01, traj_02, traj_03])
# print("Trajectories: ", trajectories.shape)

'''cut_traj list'''
# t01 = data.iloc[0:2456]
# t02 = data.iloc[2456:3960]
# t03 = data.iloc[3960:5199]
# traj_01 = generate_trajectory(t01, n_direction_bins)
# traj_02 = generate_trajectory(t02, n_direction_bins)
# traj_03 = generate_trajectory(t03, n_direction_bins)
# trajectories = [traj_01, traj_02, traj_03]
# print("Trajectories: ", len(trajectories), len(trajectories[2]), len(trajectories[1][0]))

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
discount = 0.9
# rewards = irl(feature_matrix, mdp.n_actions, mdp.discount, transition_probabilities, trajectories, epochs, learning_rate)
# rewards = maxentirl(feature_matrix, mdp.n_actions, discount, 
#                     transition_probabilities, trajectories, epochs, learning_rate)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# feature_matrix_torch = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
# transition_probabilities_torch = torch.tensor(transition_probabilities, dtype=torch.float32).to(device)
# trajectories_torch = torch.tensor(trajectories, dtype=torch.long).to(device)  # long type for indices
# rewards_torch = maxent_gpu.irl(feature_matrix_torch, mdp.n_actions, mdp.discount, 
#                                transition_probabilities_torch, trajectories_torch, 
#                                epochs, learning_rate)
# # Convert the output rewards back to a NumPy array if needed
# rewards = rewards_torch.cpu().numpy()

# #Output the inferred rewards
# print("Inferred Rewards:", rewards.shape)
# print(rewards)
# # Save the inferred rewards as a CSV file
# np.savetxt('inferred_rewards_maxent_direction.csv', rewards, delimiter=',')

# rewards = np.loadtxt('inferred_rewards.csv', delimiter=',')
# plot_grid_based_rewards(rewards, n_direction_bins, n_velocity_bins)
# visualize_rewards_heatmap(rewards, n_states, mdp.n_actions)
# plot_most_rewarded_action_heatmap(rewards, n_direction_bins=5, n_vel_bins=28)
# plot_action_reward_subplots(rewards, n_direction_bins=5, n_vel_bins=28, n_actions=6)
# plot_velocity_action_reward_heatmap(rewards, n_direction_bins=5, n_vel_bins=28)
# plot_direction_action_reward_heatmap(rewards, n_direction_bins=5, n_vel_bins=28)

from value_iteration import optimal_value

rewards = np.loadtxt('inferred_rewards_maxent_direction.csv', delimiter=',')
def find_most_rewarded_action(n_states, n_actions, transition_probabilities, reward, discount):
    """
    Find the most rewarded action for each state based on the given reward function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: NumPy array mapping (state, action, state') to the 
                              probability of transitioning from state to state' under action.
                              Shape (N, A, N).
    reward: Vector of rewards for each state. Shape (N,).
    discount: Discount factor. float.
    
    -> Array of the most rewarded actions for each state. Shape (N,).
    """
    v = optimal_value(n_states, n_actions, transition_probabilities, reward, discount)
    
    most_rewarded_actions = np.zeros(n_states, dtype=int)
    
    for s in range(n_states):
        max_q = float('-inf')
        best_action = None
        
        for a in range(n_actions):
            # Compute the Q-value for action 'a' in state 's'
            q_value = np.sum(transition_probabilities[s, a, :] * (reward + discount * v))
            
            if q_value > max_q:
                max_q = q_value
                best_action = a
                
        most_rewarded_actions[s] = best_action
    
    return most_rewarded_actions

# Assume you have already obtained the most_rewarded_actions array
most_rewarded_actions = find_most_rewarded_action(n_states, mdp.n_actions, transition_probabilities, rewards, discount)

# Reshape the array for heatmap plotting
# Assuming you want to plot a 2D grid, reshape the most_rewarded_actions array into a square shape

heatmap_data = most_rewarded_actions.reshape((n_velocity_bins, n_direction_bins))

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="viridis", cbar=True, linewidths=.5)
plt.title("Heatmap of Most Rewarded Actions at Each State")
plt.xlabel("State Grid X")
plt.ylabel("State Grid Y")
plt.show()