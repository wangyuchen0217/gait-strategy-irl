import os
import json
import numpy as np
import pandas as pd

subjects = 33
fold_path = os.getcwd() + '/expert_data_builder'

ds = None
for i in range(subjects):
    subject_number = f"{i+1:02d}"
    with open(fold_path+"/trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject_number}"]["cricket_number"]
        video_number = trail_details[f"T{subject_number}"]["video_number"]
    # read the joint movement data
    joint_path = os.path.join(fold_path, 'joint_movement', cricket_number, f'PIC{video_number}_Joint_movement.csv')
    #joint_path = fold_path + '/joint_movement/' + cricket_number + '/PIC' + video_number + '_Joint_movement.csv'
    joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0])
    if ds is None:
        ds = joint_movement
    else:
        ds = pd.concat([ds, joint_movement], axis=0, ignore_index=True)

ds.to_csv(fold_path+'/demo_dataset.csv', index=True, header=True)