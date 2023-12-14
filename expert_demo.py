from envs import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mujoco_py
from MaxEnt_IRL import MaxEntIRL
from RL import QLearningAgent

def setup_simulation(model_path):
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    return sim, viewer

def run_simulation(sim, dataset, viewer=None, learned_weights=None):
    rewards = []
    for i in range(len(dataset)):
        joint_angle = np.deg2rad(dataset[i])
        sim.data.ctrl[:] = joint_angle
        sim.step()
        # Render frames if viewer is provided
        if viewer:
            viewer.render()
        # Compute reward if reward weights are provided
        if learned_weights is not None:
            reward = learned_weights.dot(sim.get_state().qpos)
            rewards.append(reward)
    return rewards

# Load joint angle data from the CSV file
csv_file_path = 'Expert_data_builder/demo_dataset.csv'  
dataset = pd.read_csv(csv_file_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()
dataset = dataset[5200:6200, :]

#  Train: Set up simulation without rendering
sim, viewer = setup_simulation('envs/assets/Cricket2D.xml')
run_simulation(sim, dataset)

# Extract state trajectories from the simulation
state_trajectories = [sim.get_state().qpos.copy() for _ in range(len(dataset))]
state_trajectories = np.array(state_trajectories)

# Perform MaxEnt IRL training
state_dim = state_trajectories.shape[1]
epochs = 1000
irl_agent = MaxEntIRL(state_trajectories, state_dim, epochs)
learned_weights = irl_agent.maxent_irl()
irl_agent.plot_training_progress()
np.save("learned_weights.npy", learned_weights)

# Test: Set up simulation with rendering
weights = np.load("learned_weights.npy")
sim, viewer = setup_simulation('envs/assets/Cricket2D.xml')
run_simulation(sim, dataset, viewer, learned_weights)

# # Set up RL agent using Q-learning
# num_actions =  12# specify the number of actions in your environment
# rl_agent = QLearningAgent(state_dim, num_actions)

# # RL training loop
# num_episodes = 500
# for episode in range(num_episodes):
#     state = 0  # specify the initial state
#     total_reward = 0

#     while not done:  # replace with your own termination condition
#         action = rl_agent.choose_action(state)
        
#         # Apply the learned reward weights to calculate the shaped reward
#         shaped_reward = learned_weights.dot(state_trajectories[state])
        
#         next_state, reward, done, _ = env.step(action)
        
#         # Update Q-values using the shaped reward
#         rl_agent.update_q_values(state, action, shaped_reward, next_state)

#         state = next_state
#         total_reward += shaped_reward

#     print(f"Episode {episode + 1}, Total Reward: {total_reward}")



