import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py
from sklearn.preprocessing import MinMaxScaler
from algorithms.maxent_irl import MaxEntIRL
import yaml
import datetime
import dateutil.tz

def dataset_normalization(dataset):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(dataset)
    ds_scaled = scaler.transform(dataset)
    # denormalize the dataset
    # ds_rescaled = scaler.inverse_transform(ds_scaled)
    return scaler, ds_scaled

def fold_configure(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)  

# Load joint angle data from the CSV file
csv_file_path = 'expert_data_builder/demo_dataset.csv'  
dataset = pd.read_csv(csv_file_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()
#dataset = dataset[5200:6200, :]
# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)

# Get the state trajectories
trajectories = []
for i in range(len(dataset)):
    joint_angle = np.deg2rad(dataset[i])
    sim.data.ctrl[:] = joint_angle
    sim.step()
    # qpos: joint positions
    state_pos = sim.get_state().qpos.copy()
    state_vel = sim.get_state().qvel.copy()
    action = sim.data.ctrl.copy()
    trajectory = np.concatenate((state_pos, state_vel, action))
    trajectories.append(trajectory)
trajectories = np.array(trajectories)

print("trajectory", trajectories.shape)