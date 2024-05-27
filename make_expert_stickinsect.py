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

'''firl-stickinsect-v0'''
cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
                                                f"PIC{video_number}_Joint_movement.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
# joint_movement = data_smooth(joint_movement) # smooth the data

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

trajecroty = []
torso_position = []
for j in range(7100): # 7100 is the length of each trajectory

    # implement the joint angle data
    joint_angle = np.deg2rad(joint_movement[j])
    sim.data.ctrl[:] = joint_angle
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos.copy()[-12:], 
                                        sim.get_state().qvel.copy()[-12:]))
    # record the state of each step
    trajecroty.append(state) # [7100,24]
    torso_position.append(sim.data.qpos[:3].copy()) # [7100,3]

    # record the initial position
    # if j == 0:
    #     initail_pos = sim.get_state().qpos.copy()
    #     initail_pos = initail_pos[-12:]
    #     print("initail_pos:", initail_pos.shape)
    #     print("initail_pos:", initail_pos)

# record each trails
trajectories = np.array([trajecroty]) # [1, 7100, 24]
print("expert_demo:", trajectories.shape)
# np.save("Cricket2D-v1-0.01.npy", trajectories)

# record the torso position
plt.figure()
torso_position = np.array(torso_position)
plt.plot(torso_position[:,0], torso_position[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("c21_0680_trajectory_simulated")
plt.grid()
plt.show()
# plt.savefig("c21_0680_002_3.png")