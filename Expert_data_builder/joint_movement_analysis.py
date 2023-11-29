'''
This code is used to analyse the joint movement of the expert cricketers.
The minimum and maximum data of the joint movement are saved in
'Expert_data_builder/Joint_movement/joint_range_analysis.csv'
Analysis results will be used as the refernce to set the joint range of mujoco model.
'''
import os
import json
import pandas as pd

def get_joint_movement(subject:str, fold_path):
    with open(fold_path+"/trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject}"]["cricket_number"]
        video_number = trail_details[f"T{subject}"]["video_number"]
    joint_path = fold_path + '/Joint_movement/' + cricket_number + '/PIC' + video_number + '_Joint_movement.csv'
    joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0])
    df_joint = pd.DataFrame(data=joint_movement)
    return df_joint

subjects = 33
fold_path = os.getcwd() + '/Expert_data_builder'
joint_range = pd.DataFrame(columns=['ThC_LF', 'ThC_LM', 'ThC_LH', 'ThC_RF', 'ThC_RM',
                                                            'ThC_RH', 'FTi_LF', 'FTi_LM', 'FTi_LH', 'FTi_RF', 'FTi_RM', 'FTi_RH'])

for i in range(subjects):
    i = i + 1
    if i < 10:
        subject_number = "0" + str(i)
    else:
        subject_number = str(i)
    with open(fold_path+"/trail_details.json", "r") as f:
        trail_details = json.load(f)
        cricket_number =  trail_details[f"T{subject_number}"]["cricket_number"]
        video_number = trail_details[f"T{subject_number}"]["video_number"]
    # get the max and min of each joint_movement
    df_joint = get_joint_movement(subject_number, fold_path)
    min = df_joint.min()
    max = df_joint.max()
    # add the max and min to the joint_range
    joint_range.loc[cricket_number+'_'+video_number+'_'+'min'] = min
    joint_range.loc[cricket_number+'_'+video_number+'_'+'max'] = max

joint_min = joint_range.min()
joint_max = joint_range.max()
joint_range.loc['min'] = joint_min
joint_range.loc['max'] = joint_max
joint_range.to_csv(fold_path+'/Joint_movement/joint_range_analysis.csv')