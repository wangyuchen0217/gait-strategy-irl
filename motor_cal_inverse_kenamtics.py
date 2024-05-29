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

# Function to calculate torques using inverse dynamics
def calculate_inverse_dynamics_torques(sim, positions, velocities, accelerations):
    torques = []
    for qpos, qvel, qacc in zip(positions, velocities, accelerations):
        # Set the state
        sim.data.qpos[-24:] = qpos
        sim.data.qvel[-24:] = qvel
        sim.data.qacc[-24:] = qacc
        
        # Calculate inverse dynamics
        mujoco_py.functions.mj_inverse(model, sim.data)
        
        # Append the calculated torques
        torques.append(sim.data.qfrc_inverse.copy())
    
    return np.array(torques)


'''firl-stickinsect-v0'''
animal = "Carausius"
joint_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
joint_movement = data_smooth(joint_movement) # smooth the data

dt = 0.005  # The timestep of your data
# Calculate velocities and accelerations
velocities = np.diff(joint_movement, axis=0) / dt
accelerations = np.diff(velocities, axis=0) / dt
# Pad the arrays to match the length of the original data
velocities = np.vstack((velocities, np.zeros((1, velocities.shape[1]))))
accelerations = np.vstack((accelerations, np.zeros((2, accelerations.shape[1]))))

print(joint_movement.shape, velocities.shape, accelerations.shape)

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Calculate torques for the whole trajectory
torques = calculate_inverse_dynamics_torques(sim, joint_movement, velocities, accelerations)
print(torques.shape)