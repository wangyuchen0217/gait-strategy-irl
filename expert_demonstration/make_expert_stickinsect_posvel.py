import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
from envs import *
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import time
import mediapy as media
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


'''firl-stickinsect-v0'''
animal = "Carausius"
joint_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
joint_movement = data_smooth(joint_movement) # smooth the data

# FTi joint angle minus 90 degree
joint_movement[:,-6:] = joint_movement[:,-6:] - 90

dt = 0.005  # The timestep of your data
# Calculate velocities and accelerations
velocities = np.diff(joint_movement, axis=0) / dt
# Pad the arrays to match the length of the original data
velocities = np.vstack((velocities, np.zeros((1, velocities.shape[1])))) # [2459, 24]

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

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
data.qpos[-24:] = np.array(init_qpos_data.split()).astype(np.float64)

trajecroty = []
forces = []
frames = []
for j in range(2459): # 2459 is the length of each trajectory

    # implement the joint angle data
    joint_angle = np.deg2rad(joint_movement[j])
    data.ctrl[:24] = joint_angle
    data.ctrl[24:] = velocities[j]
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)

    state = np.hstack((data.qpos.copy()[:], # [-24:] joint angles, [:] w/ torso 
                                        data.qvel.copy()[:])) # [-24:] joint velocities, [:] w/ torso
    # record the state of each step
    trajecroty.append(state) # [2459,48] only joint angles and velocities, [2459, 61] w/ torso
    # get data of the torques sensor
    # forces.append(sim.data.sensordata.copy())

    # record the initial position
    if j == 0:
        initail_pos = data.qpos.copy()
        initail_pos = initail_pos[:]
        print("initail_pos:", initail_pos.shape)
        print("initail_pos:", initail_pos)

media.show_video(frames, fps=0.005)
# record each trails
trajectories = np.array([trajecroty]) # [1, 2459, 48] only joint angles and velocities, [1, 2459, 61] w/ torso
print("expert_demo:", trajectories.shape)
# np.save("StickInsect-v0.npy", trajectories)

# record the forces data
# forces = np.array(forces) # [2459, 24]
# print("forces:", forces.shape)
# forces_save_path = os.path.join("expert_data_builder/stick_insect", animal, "Animal12_110415_00_22_jointforces.csv")
# pd.DataFrame(forces).to_csv(forces_save_path, header=["LF_sup", "LM_sup", "LH_sup", "RF_sup", "RM_sup", "RH_sup",
#                                                                     "LF_CTr", "LM_CTr", "LH_CTr", "RF_CTr", "RM_CTr", "RH_CTr",
#                                                                     "LF_ThC", "LM_ThC", "LH_ThC", "RF_ThC", "RM_ThC", "RH_ThC",
#                                                                     "LF_FTi", "LM_FTi", "LH_FTi", "RF_FTi", "RM_FTi", "RH_FTi"], index=None)