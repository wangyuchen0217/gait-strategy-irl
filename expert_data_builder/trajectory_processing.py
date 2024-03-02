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

subjects = 33 
trajectories = [] # [33, 1270, 24]   
for i in range(subjects):
    subject_number = f"{i+1:02d}"
    with open("expert_data_builder/trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject_number}"]["cricket_number"]
        video_number = trail_details[f"T{subject_number}"]["video_number"]
    # read the joint movement data
    csv_file_path = os.path.join("expert_data_builder/velocity_data", cricket_number, 
                                                f"{video_number}_Velocity_Smooth.csv")
    vel = pd.read_csv(csv_file_path, header=None, index_col=[1,2]).to_numpy() # vel.x and vel.y
    