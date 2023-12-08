import gym
from envs import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mujoco_py
from MaxEnt_IRL import MaxEntIRL

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
        
        if viewer:
            viewer.render()
            
        if learned_weights is not None:
            reward = learned_weights.dot(sim.get_state().qpos)
            rewards.append(reward)
    
    return rewards

# Load joint angle data from the CSV file
csv_file_path = 'Expert_data_builder/demo_dataset.csv'  
dataset = pd.read_csv(csv_file_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()
dataset = dataset[5200:6200, :]

# Set up simulation without rendering
sim, _ = setup_simulation('envs/assets/Cricket2D.xml')

# Apply joint angles from the CSV data to the MuJoCo model
for i in range(len(dataset)):
    joint_angle = np.deg2rad(dataset[i])
    sim.data.ctrl[:] = joint_angle
    sim.step()

# Extract state trajectories from the simulation
state_trajectories = [sim.get_state().qpos.copy() for _ in range(len(dataset))]
state_trajectories = np.array(state_trajectories)

# Perform MaxEnt IRL training
state_dim = state_trajectories.shape[1]
epochs = 1000
irl_agent = MaxEntIRL(state_trajectories, state_dim, epochs)
irl_agent.maxent_irl()

# Access the learned reward weights and plot training progress
learned_weights = irl_agent.get_learned_weights()
irl_agent.plot_training_progress()

# Run the simulation again with the learned reward weights and render frames
sim, viewer = setup_simulation('envs/assets/Cricket2D.xml')
rewards = run_simulation(sim, dataset, viewer, learned_weights)

# Print rewards
# for i, reward in enumerate(rewards):
#     print(f"Step {i+1}: Reward = {reward}")
# plot the reward
plt.plot(rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward vs. Step")
plt.show()
plt.savefig("reward.png")


