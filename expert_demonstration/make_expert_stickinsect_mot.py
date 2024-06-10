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


'''firl-stickinsect-v1'''
animal = "Carausius"
forces_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22_forces.csv")
forces = pd.read_csv(forces_path, header=[0], index_col=None).to_numpy()
forces_unsmoothed = forces.copy()
forces = data_smooth(forces) # smooth the data

# calcuate the torque data
leg_lengths = np.array([0.13, 0.14, 0.15, 0.13, 0.14, 0.15, 
                        3.21, 2.42, 2.95, 3.21, 2.42, 2.95, 
                        1.58, 1.16, 1.39, 1.58, 1.16, 1.39, 
                        1.5, 1.12, 1.41, 1.5, 1.12, 1.41])
torques = np.zeros(forces.shape)
for i in range(len(forces)):
    torques[i] = forces[i] * leg_lengths
print("torques:", torques.shape)

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

# Define PD controller parameters
Kp = 0.5  # Proportional gain
Kd = 0.1  # Derivative gain
target_positions = np.zeros(24)  # Assuming 24 joints, adjust as necessary

trajecroty = []
torq= []
pd_torque = []
for j in range(2459): # 2459 is the length of each trajectory
    current_positions = sim.data.qpos[-24:]  # Current joint positions
    current_velocities = sim.data.qvel[-24:]  # Current joint velocities
    position_error = target_positions - current_positions  # Position error
    velocity_error = -current_velocities  # Velocity error (damping term)

    # Calculate torques using PD control
    pd_torques = Kp * position_error + Kd * velocity_error
    total_torques = torques[j] + pd_torques

    # implement the motor data
    sim.data.ctrl[:] = total_torques
    sim.step()
    # viewer.render()
    state = np.hstack((sim.get_state().qpos.copy()[-24:], 
                                        sim.get_state().qvel.copy()[-24:]))
    # record the state of each step
    trajecroty.append(state) # [2459,24]
    torq.append(total_torques) # [2459,24]
    pd_torque.append(pd_torques) # [2459,24]

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

# subplot the torque data and torq
torq = np.array(torq)
pd_torques = np.array(pd_torque)
fig, axs = plt.subplots(3, 1, figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
axs[0].plot(torques[:,0])
axs[0].plot(torques[:,6])
axs[0].plot(torques[:,12])
axs[0].plot(torques[:,18])
axs[0].set_xlabel('Frame', fontsize=14)
axs[0].set_ylabel('Torques', fontsize=14)
axs[0].set_title('Carausius_110415_00_22_torques', fontsize=14)
axs[0].grid()

axs[1].plot(torq[:,0])
axs[1].plot(torq[:,6])
axs[1].plot(torq[:,12])
axs[1].plot(torq[:,18])
axs[1].set_xlabel('Frame', fontsize=14)
axs[1].set_ylabel('Torq', fontsize=14)
axs[1].set_title('Carausius_110415_00_22_torq', fontsize=14)
axs[1].grid()

axs[2].plot(pd_torques[:,0])
axs[2].plot(pd_torques[:,6])
axs[2].plot(pd_torques[:,12])
axs[2].plot(pd_torques[:,18])
axs[2].set_xlabel('Frame', fontsize=14)
axs[2].set_ylabel('PD_Torques', fontsize=14)
axs[2].set_title('Carausius_110415_00_22_pd_torques', fontsize=14)
axs[2].grid()
plt.show()


# plot and compare the data
# joint_path = os.path.join("expert_data_builder/stick_insect", animal,   
#                                                 "Animal12_110415_00_22.csv")
# joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()

# fig, axs = plt.subplots(3, 1, figsize=(15, 10))
# plt.subplots_adjust(hspace=0.5)
# axs[0].plot(joint_movement[:,12])
# axs[0].set_xlabel('Frame', fontsize=14)
# axs[0].set_ylabel('Joint Movement', fontsize=14)
# axs[0].set_title('Carausius_110415_00_22_joint_movement', fontsize=14)
# axs[0].grid()

# axs[1].plot(forces[:,12])
# axs[1].set_xlabel('Frame', fontsize=14)
# axs[1].set_ylabel('Forces_smooth', fontsize=14)
# axs[1].set_title('Carausius_110415_00_22_forces_smooth', fontsize=14)
# axs[1].grid()

# axs[2].plot(forces_unsmoothed[:,12])
# axs[2].set_xlabel('Frame', fontsize=14)
# axs[2].set_ylabel('Forces', fontsize=14)
# axs[2].set_title('Carausius_110415_00_22_forces', fontsize=14)
# axs[2].grid()
# plt.savefig("Carausius_110415_00_22.png")