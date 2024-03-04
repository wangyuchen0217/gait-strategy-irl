import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py
import yaml
import json
import matplotlib.pyplot as plt

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

'''let's do irl'''
# # Get the state trajectories
# subjects = 33 
# trajectories = [] # [33, 1270, 2, 12]   
# for i in range(subjects):
#     subject_number = f"{i+1:02d}"
#     with open("expert_data_builder/trail_details.json", "r") as f:
#         trail_details = json.load(f)
#         cricket_number =  trail_details[f"T{subject_number}"]["cricket_number"]
#         video_number = trail_details[f"T{subject_number}"]["video_number"]
#     # read the joint movement data
#     csv_file_path = os.path.join("expert_data_builder/joint_movement", cricket_number, 
#                                                 f"PIC{video_number}_Joint_movement.csv")
#     trail = pd.read_csv(csv_file_path, header=[0], 
#                                         index_col=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()
#     trajecroty = []
#     for j in range(1270): # 1270 is the length of each trajectory
#         joint_angle = np.deg2rad(trail[j])
#         sim.data.ctrl[:] = joint_angle
#         sim.step()
#         viewer.render()
#         state = sim.get_state().qpos.copy()
#         action = sim.data.ctrl.copy()
#         # record the state and action of each step
#         traj_step = np.array((state, action)) # [2, 12]
#         trajecroty.append(traj_step) # [1270, 2, 12]
#     # record each trails
#     trajectories.append(trajecroty) # [33, 1270, 2, 12]
# trajectories = np.array(trajectories)
# print("expert_demo:", trajectories.shape)
# np.save("expert_demo.npy", trajectories)

'''firl-2d resting position'''
# subjects = 33 
# trajectories = [] # [33, 1270, 24]   
# for i in range(subjects):
#     subject_number = f"{i+1:02d}"
#     with open("expert_data_builder/trail_details.json", "r") as f:
#         trail_details = json.load(f)
#         cricket_number =  trail_details[f"T{subject_number}"]["cricket_number"]
#         video_number = trail_details[f"T{subject_number}"]["video_number"]
#     # read the joint movement data
#     csv_file_path = os.path.join("expert_data_builder/joint_movement", cricket_number, 
#                                                 f"PIC{video_number}_Joint_movement.csv")
#     trail = pd.read_csv(csv_file_path, header=[0], index_col=[0]).to_numpy()
#     trajecroty = []
#     for j in range(1270): # 1270 is the length of each trajectory
#         joint_angle = np.deg2rad(trail[j])
#         sim.data.ctrl[:] = joint_angle
#         sim.step()
#         #viewer.render()
#         state = np.hstack((sim.get_state().qpos.copy(), 
#                                             sim.get_state().qvel.copy()))
#         # record the state of each step
#         trajecroty.append(state) # [1270, 24]
#     # record each trails
#     trajectories.append(trajecroty) # [33, 1270, 24]
# trajectories = np.array(trajectories)
# print("expert_demo:", trajectories.shape)
# np.save("CricketEnv2D-v0.npy", trajectories)

'''firl-2d moving position'''
cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
                                                f"PIC{video_number}_Joint_movement.csv")
direction_path = os.path.join("expert_data_builder/movement", cricket_number,
                                                f"PIC{video_number}_Heading_direction.csv")
traj_path = os.path.join("expert_data_builder/movement", cricket_number,
                                                f"PIC{video_number}_Trajectory.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
heading_direction = pd.read_csv(direction_path, header=[0], index_col=[0]).to_numpy()
traj = pd.read_csv(traj_path, header=[0], index_col=[0]).to_numpy() # traj.x and traj.y
# traj scale
traj = traj * 100 # original measurement is in meters, m -> cm
trajecroty = []
for i in range(7100): # 7100 is the length of each trajectory
    joint_angle = np.deg2rad(joint_movement[i])
    direction = np.deg2rad(heading_direction[i])
    sim.data.ctrl[:12] = joint_angle
    sim.data.ctrl[12:14] = traj[i, :]
    sim.data.ctrl[14] = direction
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos[:12].copy(), 
                                        sim.get_state().qvel[:12].copy()))
    # record the state of each step
    trajecroty.append(state) # [7100, 24]
trajectories = np.array([trajecroty]) # [1, 7100, 24]
print("expert_demo:", trajectories.shape)
#np.save("CricketEnv2D-v0.npy", trajectories)
