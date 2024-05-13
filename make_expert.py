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

# gait phase definition
def gait_phase(gait_signals):
    if gait_signals == 1: # stance phase
        friction = [100, 0.5, 0.5]
    else:
        friction = [0, 0, 0]
    return friction

'''firl-2d-v1'''
cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
                                                f"PIC{video_number}_Joint_movement.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
joint_movement = data_smooth(joint_movement) # smooth the data

# set up the gait phase
gait_path = os.path.join("expert_data_builder/movement", cricket_number,
                                                f"PIC{video_number}_Gait_phase.csv")
gait = pd.read_csv(gait_path, header=[0], index_col=[0]).to_numpy()
gait = gait[:,:6] # only use the ThC joint to define the gait phase

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Set the initial positions of the legs
initial_leg_positions = config_data.get("initial_leg_positions")
print("initial_leg_positions:", initial_leg_positions)
for i, idx in enumerate(["LF_hip", "LF_knee", "RF_hip", "RF_knee",
                        "LM_hip", "LM_knee", "RM_hip", "RM_knee",
                        "LH_hip", "LH_knee", "RH_hip", "RH_knee"]):
    sim.data.qpos[model.get_joint_qpos_addr(idx)] = initial_leg_positions[i]

LF_tip_idx = sim.model.geom_name2id('LF_tip_geom')
RF_tip_idx = sim.model.geom_name2id('RF_tip_geom')
LM_tip_idx = sim.model.geom_name2id('LM_tip_geom')
RM_tip_idx = sim.model.geom_name2id('RM_tip_geom')
LH_tip_idx = sim.model.geom_name2id('LH_tip_geom')
RH_tip_idx = sim.model.geom_name2id('RH_tip_geom')

trajecroty = []
for j in range(7100): # 7100 is the length of each trajectory

    # implement the gait phase data
    gait_signals = gait[j] # [6,]
    for i, idx in enumerate([LF_tip_idx, RF_tip_idx]):
        gait_data = gait_signals[i]
        sim.model.geom_friction[idx, :] = gait_phase(gait_data)
    for i, idx in enumerate([LM_tip_idx, RM_tip_idx]):
        gait_data = gait_signals[i+2]* 7
        sim.model.geom_friction[idx, :] = gait_phase(gait_data)
    for i, idx in enumerate([LH_tip_idx, RH_tip_idx]):
        gait_data = gait_signals[i+4]
        sim.model.geom_friction[idx, :] = gait_phase(gait_data)

    # implement the joint angle data
    joint_angle = np.deg2rad(joint_movement[j])
    sim.data.ctrl[:12] = joint_angle
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos.copy()[-12:], 
                                        sim.get_state().qvel.copy()[-12:]))
    # record the state of each step
    trajecroty.append(state) # [7100,24]

    # # record the initial position
    # if j == 0:
    #     initail_pos = sim.get_state().qpos.copy()
    #     initail_pos = initail_pos[-12:]
    #     print("initail_pos:", initail_pos.shape)
    #     print("initail_pos:", initail_pos)

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
