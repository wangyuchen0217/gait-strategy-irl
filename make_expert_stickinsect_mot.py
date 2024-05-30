import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


'''firl-stickinsect-v1'''
animal = "Carausius"
forces_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22_forces.csv")
forces = pd.read_csv(forces_path, header=[0], index_col=None).to_numpy()
forces = data_smooth(forces) # smooth the data

# calcuate the torque data


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
torso_position = []
for j in range(2459): # 2459 is the length of each trajectory

    # implement the motor data
    sim.data.ctrl[:] = forces[j]
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos.copy()[-24:], 
                                        sim.get_state().qvel.copy()[-24:]))
    # record the state of each step
    trajecroty.append(state) # [2459,24]
    torso_position.append(sim.data.qpos[:3].copy()) # [2459,3]

    # record the initial position
    if j == 0:
        initail_pos = sim.get_state().qpos.copy()
        initail_pos = initail_pos[:]
        print("initail_pos:", initail_pos.shape)
        print("initail_pos:", initail_pos)

# record each trails
trajectories = np.array([trajecroty]) # [1, 2459, 24]
print("expert_demo:", trajectories.shape)
# np.save("StickInect-v0.npy", trajectories)

# record the torso position
# plt.figure()
# torso_position = np.array(torso_position)
# plt.plot(torso_position[:,0], torso_position[:,1])
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("c21_0680_trajectory_simulated")
# plt.grid()
# plt.show()
# # plt.savefig("c21_0680_002_3.png")