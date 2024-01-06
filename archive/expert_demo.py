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
pd.DataFrame(trajectories).to_csv("state_trajectories.csv", 
                                                                                    header=None, index=None)
# Normalize the state trajectories
scaler, state_trajectories = dataset_normalization(trajectories)

# Perform MaxEnt IRL training
state_dim = state_trajectories.shape[1]
epochs = config_data.get("epochs")
learning_rate = config_data.get("learning_rate")
irl_agent = MaxEntIRL(state_trajectories, state_dim, epochs, learning_rate)
learned_weights = irl_agent.maxent_irl()

# logs
# make a folder under logs named current env name
env = config_data.get("env")
exp_id = os.path.join('logs', env)
fold_configure(exp_id)
now = datetime.datetime.now(dateutil.tz.tzlocal())
log_folder = exp_id + "/" + now.strftime('%Y%m%d_%H%M')
fold_configure(log_folder)
# copy the config file 
os.system("cp configs/irl.yml " + log_folder + "/config.yml")
# save the training progress
irl_agent.plot_training_progress(log_folder + "/training_progress.png")
# save the reward history
irl_agent.save_reward_history(log_folder + "/reward_history.csv")
# save the learned weights
irl_agent.save_learned_weights(log_folder + "/learned_weights.csv", format="csv")
irl_agent.save_learned_weights(log_folder + "/learned_weights.npy", format="npy")
