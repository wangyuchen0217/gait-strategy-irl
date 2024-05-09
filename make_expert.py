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

'''firl-2d resting position'''
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
for j in range(7100): # 7100 is the length of each trajectory
    # implement a vitual force on legs
    xfrc_applied = sim.data.xfrc_applied # [26, 6] is the force applied to each body
    body_idx = sim.model.body_name2id('RH_tip') # idx starts from 0
    force = np.array([-5000, 100000, 0])
    sim.data.xfrc_applied[body_idx, :3] = force

    # implement the joint angle data
    joint_angle = np.deg2rad(joint_movement[j])
    # sim.data.ctrl[:] = joint_angle
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos.copy(), 
                                        sim.get_state().qvel.copy()))
    # record the state of each step
    trajecroty.append(state) # [7100,24]
# record each trails
trajectories = np.array([trajecroty]) # [1, 7100, 24]
print("expert_demo:", trajectories.shape)
# np.save("CricketEnv2D-v0.npy", trajectories)

'''firl-3d  ThC joint smoothed data position'''
# cricket_number = 'c21'
# video_number = '0680'
# joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
#                                                 f"PIC{video_number}_Joint_movement.csv")
# joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
# joint_movement = data_smooth(joint_movement) # smooth the data
# joint_movement = joint_movement*2 # scale the data

# #  Set up simulation without rendering
# model_name = config_data.get("model")
# model_path = 'envs/assets/' + model_name + '.xml'
# model = mujoco_py.load_model_from_path(model_path)
# sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)

# trajecroty = []
# for i in range(7100): # 7100 is the length of each trajectory
#     joint_angle = np.deg2rad(joint_movement[i])
#     sim.data.ctrl[:6] = joint_angle[:6] # ThC joint only
#     sim.step()
#     viewer.render()
#     # record the state
#     state = np.hstack((sim.get_state().qpos[:].copy(), 
#                                         sim.get_state().qvel[:].copy()))
#     trajecroty.append(state) # [7100, 24]
# trajectories = np.array([trajecroty]) # [1, 7100, 24]
# print("expert_demo:", trajectories.shape)
# # np.save("CricketEnv2D-v0-moving-torso.npy", trajectories)

'''firl-3d  ThC joint smoothed data motor'''
# cricket_number = 'c21'
# video_number = '0680'
# joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
#                                                 f"PIC{video_number}_Joint_movement.csv")
# joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
# joint_movement = data_smooth(joint_movement) # smooth the data
# joint_movement = normalize(joint_movement)

# #  Set up simulation without rendering
# model_name = config_data.get("model")
# model_path = 'envs/assets/' + model_name + '.xml'
# model = mujoco_py.load_model_from_path(model_path)
# sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)

# trajecroty = []
# for i in range(7100): # 7100 is the length of each trajectory
#     joint_angle = joint_movement[i]
#     sim.data.ctrl[:6] = joint_angle[:6] # ThC joint only
#     sim.step()
#     viewer.render()
#     # record the state
#     state = np.hstack((sim.get_state().qpos[:].copy(), 
#                                         sim.get_state().qvel[:].copy()))
#     trajecroty.append(state) # [7100, 24]
# trajectories = np.array([trajecroty]) # [1, 7100, 24]
# print("expert_demo:", trajectories.shape)
# # np.save("CricketEnv2D-v0-moving-torso.npy", trajectories)

'''firl-2d moving position'''
# cricket_number = 'c21'
# video_number = '0680'
# joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
#                                                 f"PIC{video_number}_Joint_movement.csv")
# direction_path = os.path.join("expert_data_builder/movement", cricket_number,
#                                                 f"PIC{video_number}_Heading_direction.csv")
# traj_path = os.path.join("expert_data_builder/movement", cricket_number,
#                                                 f"PIC{video_number}_Trajectory.csv")
# joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
# heading_direction = pd.read_csv(direction_path, header=[0], index_col=[0]).to_numpy()
# traj = pd.read_csv(traj_path, header=[0], index_col=[0]).to_numpy() # traj.x and traj.y
# # traj scale
# traj = traj * 100 # original measurement is in meters m->cm
# trajecroty = []
# for i in range(7100): # 7100 is the length of each trajectory
#     joint_angle = np.deg2rad(joint_movement[i])
#     direction = np.deg2rad(heading_direction[i])
#     sim.data.ctrl[:12] = joint_angle
#     sim.data.ctrl[12:14] = traj[i, :]
#     sim.data.ctrl[14] = direction
#     sim.step()
#     viewer.render()
#     # record the state
#     state = np.hstack((sim.get_state().qpos[:].copy(), 
#                                         sim.get_state().qvel[:].copy()))
#     trajecroty.append(state) # [7100, 24]
# trajectories = np.array([trajecroty]) # [1, 7100, 24]
# print("expert_demo:", trajectories.shape)
# np.save("CricketEnv2D-v0-moving-torso.npy", trajectories)

'''firl-2d moving gait phase'''
# cricket_number = 'c21'
# video_number = '0680'
# gait_path = os.path.join("expert_data_builder/movement", cricket_number,
#                                                 f"PIC{video_number}_Gait_phase.csv")
# direction_path = os.path.join("expert_data_builder/movement", cricket_number,
#                                                 f"PIC{video_number}_Heading_direction.csv")
# traj_path = os.path.join("expert_data_builder/movement", cricket_number,
#                                                 f"PIC{video_number}_Trajectory.csv")
# gait = pd.read_csv(gait_path, header=[0], index_col=[0]).to_numpy()
# heading_direction = pd.read_csv(direction_path, header=[0], index_col=[0]).to_numpy()
# traj = pd.read_csv(traj_path, header=[0], index_col=[0]).to_numpy() # traj.x and traj.y
# # traj scale
# traj = traj * 100 # original measurement is in meters m->cm
# trajecroty = []
# for i in range(7100): # 7100 is the length of each trajectory
#     sim.data.ctrl[:12] = gait[i, :]
#     sim.data.ctrl[12:14] = traj[i, :]
#     sim.data.ctrl[14] = heading_direction[i]
#     sim.step()
#     viewer.render()
#     # record the state
#     state = np.hstack((sim.get_state().qpos[:].copy(), 
#                                         sim.get_state().qvel[:].copy()))
#     trajecroty.append(state) # [7100, 24]
# trajectories = np.array([trajecroty]) # [1, 7100, 24]
# print("expert_demo:", trajectories.shape)
# # np.save("CricketEnv2D-v0-gait.npy", trajectories)
