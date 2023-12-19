import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py
from sklearn.preprocessing import MinMaxScaler
from irl.maxent_irl import MaxEntIRL

csv_file_path = 'Expert_data_builder/demo_dataset.csv'  
dataset = pd.read_csv(csv_file_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()

learned_weights = np.load("learned_weights.npy")
model_path = 'envs/assets/Cricket2D.xml'

model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
#viewer = mujoco_py.MjViewer(sim)
rewards = []
for i in range(len(dataset)):
    joint_angle = np.deg2rad(dataset[i])
    sim.data.ctrl[:] = joint_angle
    sim.step()
    #viewer.render()
    # reward = learned_weights.dot(sim.get_state().qpos)
    reward = sim.get_state().qpos.copy()
    rewards.append(reward)
rewards = np.array(rewards)
pd.DataFrame(rewards).to_csv("rewards.csv", header=None, index=None)