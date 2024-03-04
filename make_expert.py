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
def trajectory_calculation(vel):
    # vel: [vel.x, vel.y]
    # scale velocity: mm/s
    vel = vel * 0.224077
    # frequency: 119.88(120)Hz
    # trajectory calculation
    traj_x = [0]; x=0
    traj_y = [0]; y=0
    for i in range(1, len(vel)):
        x = x + vel[i][0]*1/119.88/100 # dm
        y = y + vel[i][1]*1/119.88/100 # dm
        traj_x.append(x)
        traj_y.append(y)
    traj = np.array([traj_x, traj_y]).reshape(-1, 2)
    return traj

cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
                                                f"PIC{video_number}_Joint_movement.csv")
direction_path = os.path.join("expert_data_builder/movement", cricket_number,
                                                f"PIC{video_number}_Heading_direction.csv")
vel_path = os.path.join("expert_data_builder/velocity_data", cricket_number, 
                                                f"{video_number}_Velocity_Smooth.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
heading_direction = pd.read_csv(direction_path, header=[0], index_col=[0]).to_numpy()
vel = pd.read_csv(vel_path, header=None, usecols=[1,2]).to_numpy() # vel.x and vel.y
traj = trajectory_calculation(vel)
trajecroty = []
for i in range(7100): # 7100 is the length of each trajectory
    joint_angle = np.deg2rad(joint_movement[i])
    direction = np.deg2rad(heading_direction[i])
    #sim.data.ctrl[:12] = joint_angle
    #sim.data.ctrl[12] = traj[i, 0]
    sim.data.ctrl[13] = traj[i, 1]
    #sim.data.ctrl[14] = direction
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos[:12].copy(), 
                                        sim.get_state().qvel[:12].copy()))
    # record the state of each step
    trajecroty.append(state) # [7100, 24]
trajectories = np.array([trajecroty]) # [1, 7100, 24]
print("expert_demo:", trajectories.shape)
#np.save("CricketEnv2D-v0.npy", trajectories)
