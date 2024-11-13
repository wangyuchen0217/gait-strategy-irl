import os
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from plot_expert import *
import yaml
import expert_data_builder.stick_insect.open_source_data.antenna_phase_encode as ape


def get_cont_data(subject:str, trim=False, trim_len:int=0):
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)
    insect_name = trail_details[f"T{subject}"]["insect_name"]
    insect_number = trail_details[f"T{subject}"]["insect_number"]
    id_1 = trail_details[f"T{subject}"]["id_1"]
    id_2 = trail_details[f"T{subject}"]["id_2"]
    id_3 = trail_details[f"T{subject}"]["id_3"]
    vel_path = os.path.join("expert_data_builder/stick_insect", insect_name,
                                                    f"{insect_number}_{id_1}_{id_2}_{id_3}_vel.csv")
    direction_path = os.path.join("expert_data_builder/stick_insect", insect_name,
                                                    f"{insect_number}_{id_1}_{id_2}_{id_3}_direction.csv")
    gait_path = os.path.join("expert_data_builder/stick_insect", insect_name,
                                                    f"{insect_number}_{id_1}_{id_2}_{id_3}_gait.csv")
    # the first value of vel and the last value of gait are not valid
    vel = pd.read_csv(vel_path, header=[0], index_col=None).to_numpy()[1:-1]
    direction = pd.read_csv(direction_path, header=[0], index_col=None).to_numpy()[1:-1]
    gait = pd.read_csv(gait_path, header=[0], index_col=None).to_numpy()[1:-1]
    if trim:
        vel = vel[:-trim_len]
        direction = direction[:-trim_len]
        gait = gait[:-trim_len]
    return vel, direction, gait

def load_antenna_dist(subject:str):
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)
    insect_name = trail_details[f"T{subject}"]["insect_name"]
    insect_number = trail_details[f"T{subject}"]["insect_number"]
    id_1 = trail_details[f"T{subject}"]["id_1"]
    id_2 = trail_details[f"T{subject}"]["id_2"]
    id_3 = trail_details[f"T{subject}"]["id_3"]
    antenna_path = os.path.join("expert_data_builder/stick_insect", insect_name,
                                                    f"{insect_number}_{id_1}_{id_2}_{id_3}_antenna_dist.csv")
    antenna = pd.read_csv(antenna_path, header=[0], index_col=None).to_numpy()
    return antenna


with open('configs/irl.yml') as file:
    v = yaml.load(file, Loader=yaml.FullLoader)

# Load the dataset
data_source = str(v['data_source'])
No1, No2, No3 = v[data_source]['No1'], v[data_source]['No2'], v[data_source]['No3']

# When the data source is [one insect]
if data_source == 'MedauroideaC00T':
    vel_01, direction_01, gait_01 = get_cont_data(No1, trim=True, trim_len=800)
    vel_02, direction_02, gait_02 = get_cont_data(No2, trim=True, trim_len=2200)
    vel_03, direction_03, gait_03 = get_cont_data(No3, trim=True, trim_len=1600)
else:
    vel_01, direction_01, gait_01 = get_cont_data(No1)
    vel_02, direction_02, gait_02 = get_cont_data(No2)
    vel_03, direction_03, gait_03 = get_cont_data(No3)
vel = np.concatenate((vel_01[1:], vel_02[1:], vel_03[1:]), axis=0)
direction = np.concatenate((direction_01[1:], direction_02[1:], direction_03[1:]), axis=0)
gait = np.concatenate((gait_01[1:], gait_02[1:], gait_03[1:]), axis=0)
print("length of T"+No1+", T"+No2+", T"+No3+": ", len(gait_01), len(gait_02), len(gait_03))
print("length of faltten trajectory:", len(gait))

# get the binned antenna data
antenna_dist_01 = ape.get_antenna_dist(No1, bin_step=60)
antenna_dist_02 = ape.get_antenna_dist(No2, bin_step=60)
antenna_dist_03 = ape.get_antenna_dist(No3, bin_step=60)
# the first value of vel and the last value of gait are not valid, acc calculation length is 1 less due to diff
antenna_dist = np.concatenate((antenna_dist_01[2:-1], antenna_dist_02[2:-1], antenna_dist_03[2:-1]), axis=0)
HS_left, HS_right, SP_left, SP_right = antenna_dist[:, 0], antenna_dist[:, 1], antenna_dist[:, 2], antenna_dist[:, 3]
print("length of antennae: ", len(HS_left))

# Load grouped gait combinations (6 types)
grouped_gait_combinations = v['grouped_gait_combinations']
# Combine the first six columns into a string for each row to represent the gait pattern
gait_data = pd.DataFrame(gait)
gait_data['Gait Pattern'] = gait_data.iloc[:, :6].astype(str).agg(''.join, axis=1)
# Categorize each gait pattern based on the possible combinations
gait_data['Category'] = gait_data['Gait Pattern'].map(grouped_gait_combinations).astype(int)
# Display the categorized data
print(gait_data[['Gait Pattern', 'Category']])

# Combine antenna data and gait pattern into a single DataFrame for analysis
analysis_df = pd.DataFrame({
        'HS left': HS_left,
        'HS right': HS_right,
        'SP left': SP_left,
        'SP right': SP_right,
        'Gait Category': gait_data['Category']
    })

save = True
if save:
    save_path = 'expert_demonstration/expert/'+data_source+'.csv'
    analysis_df.to_csv(save_path, index=False, header=True)