import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py
import yaml
import json

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)

# Get the state trajectories
subjects = 33 
trajectories = [] # [33, 1270, 3, 12]   
for i in range(subjects):
    subject_number = f"{i+1:02d}"
    with open("expert_data_builder/trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject_number}"]["cricket_number"]
        video_number = trail_details[f"T{subject_number}"]["video_number"]
    # read the joint movement data
    csv_file_path = os.path.join("expert_data_builder/joint_movement", cricket_number, 
                              f"PIC{video_number}_Joint_movement.csv")
    trail = pd.read_csv(csv_file_path, header=[0], index_col=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()
    trajecroty = []
    for j in range(1270): # 1270 is the length of each trajectory
        joint_angle = np.deg2rad(trail[j])
        sim.data.ctrl[:] = joint_angle
        sim.step()
        state = sim.get_state().qpos.copy()
        action = sim.data.ctrl.copy()
        # record the state and action of each step
        traj_step = np.array((state, action)) # [2, 12]
        trajecroty.append(traj_step) # [1270, 2, 12]
    # record each trails
    trajectories.append(trajecroty) # [33, 1270, 2, 12]
trajectories = np.array(trajectories)
print("expert_demo:", trajectories.shape)
np.save("expert_demo.npy", trajectories)
