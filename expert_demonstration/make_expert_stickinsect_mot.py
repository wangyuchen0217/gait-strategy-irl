import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
from envs import *
import numpy as np
import pandas as pd
import mujoco_py    
import yaml
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# normalization
def data_scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled

def normalize(data):
    for i in range(data.shape[1]):
        data[:,i] = data_scale(data[:,i].reshape(-1,1)).reshape(-1)
    return data

# smooth the data
def Kalman1D(observations,damping=1):
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.03
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
            )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def data_smooth(data):
    for i in range(data.shape[1]):
        smoothed_data = Kalman1D(data[:,i], damping=1).reshape(-1,1)
        data[:,i] = smoothed_data[:,0]
    return data


'''actuatorfrc'''
# animal = "Carausius"
# forces_path = os.path.join("expert_data_builder/stick_insect", animal, 
#                                                 "Animal12_110415_00_22_forces.csv")
# forces = pd.read_csv(forces_path, header=[0], index_col=None).to_numpy()
# forces_unsmoothed = forces.copy()
# forces = data_smooth(forces) # smooth the data

# # calcuate the torque data
# leg_lengths = np.array([0.13, 0.14, 0.15, 0.13, 0.14, 0.15, 
#                         3.21, 2.42, 2.95, 3.21, 2.42, 2.95, 
#                         1.58, 1.16, 1.39, 1.58, 1.16, 1.39, 
#                         1.5, 1.12, 1.41, 1.5, 1.12, 1.41])
# torques = np.zeros(forces.shape)
# for i in range(len(forces)):
#     torques[i] = forces[i] * leg_lengths
# print("torques:", torques.shape)

'''torque'''
animal = "Carausius"
torques_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22_torques_1.csv")
torques = pd.read_csv(torques_path, header=[0], index_col=None).to_numpy()
print("torques:", torques.shape)

# # convert the generalized data to scalar values
# torque_scalars = np.zeros((torques.shape[0], torques.shape[1]//3))
# for i in range(torques.shape[0]):
#     for j in range(0, torques.shape[1], 3):
#         idx = j//3
#         torque_scalars[i,idx]= np.sqrt(torques[i,j]**2 + torques[i,j+1]**2 + torques[i,j+2]**2)
# print("torque_scalars:", torque_scalars.shape)

torques_unsmoothed = torques.copy()
torques = data_smooth(torques) # smooth the data
# torques_scalar_unsmoothed = torque_scalars.copy()
# torques_scalar = data_smooth(torque_scalars) # smooth the data

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Parse the XML file to extract custom data
tree = ET.parse(model_path)
root = tree.getroot()
# Find the custom element and extract the init_qpos data
init_qpos_data = None
for custom in root.findall('custom'):
    for numeric in custom.findall('numeric'):
        if numeric.get('name') == 'init_qpos':
            init_qpos_data = numeric.get('data')
            break
sim.data.qpos[-24:] = np.array(init_qpos_data.split()).astype(np.float64)

trajecroty = []
for j in range(2459): # 2459 is the length of each trajectory

    # implement the motor data
    sim.data.ctrl[:] = torques[j]
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos.copy()[-24:], 
                                        sim.get_state().qvel.copy()[-24:]))
    # record the state of each step
    trajecroty.append(state) # [2459,24]

    # record the initial position
    if j == 0:
        initail_pos = sim.get_state().qpos.copy()
        initail_pos = initail_pos[:]
        print("initail_pos:", initail_pos.shape)
        print("initail_pos:", initail_pos)

# record each trails
trajectories = np.array([trajecroty]) # [1, 2459, 24]
print("expert_demo:", trajectories.shape)
# np.save("StickInsect-v0.npy", trajectories)
