import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py
from sklearn.preprocessing import MinMaxScaler
from MaxEnt_IRL import MaxEntIRL

def dataset_normalization(dataset):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(dataset)
    ds_scaled = scaler.transform(dataset)
    # denormalize the dataset
    # ds_rescaled = scaler.inverse_transform(ds_scaled)
    return scaler, ds_scaled

def setup_simulation(model_path):
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    # viewer = mujoco_py.MjViewer(sim)
    return sim

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
scaler, ds_scaled = dataset_normalization(dataset)
#dataset = dataset[5200:6200, :]

#  Train: Set up simulation without rendering
model_path = 'envs/assets/Cricket2D.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)
state_trajectories = []
for i in range(len(dataset)):
    joint_angle = np.deg2rad(dataset[i])
    sim.data.ctrl[:] = joint_angle
    sim.step()
    state_trajectory = sim.get_state().qpos.copy()
    state_trajectories.append(state_trajectory)
pd.DataFrame(state_trajectories).to_csv("state_trajectories.csv", header=None, index=None)

# Perform MaxEnt IRL training
# state_dim = state_trajectories.shape[1]
# epochs = 100
# irl_agent = MaxEntIRL(state_trajectories, state_dim, epochs)
# learned_weights = irl_agent.maxent_irl()
# irl_agent.plot_training_progress()
# np.save("learned_weights.npy", learned_weights)

# Test: Set up simulation with rendering
# learned_weights = np.load("learned_weights.npy")
# sim = setup_simulation('envs/assets/Cricket2D.xml')
# viewer = mujoco_py.MjViewer(sim)
# run_simulation(sim, dataset, viewer, learned_weights)





