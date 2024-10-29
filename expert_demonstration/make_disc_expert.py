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

def calculate_acceleration(vel):
    # frequency = 200 Hz
    acc = np.diff(vel, axis=0) / 0.005
    return acc

with open("configs/irl.yml", "r") as f:
    irl_config = yaml.safe_load(f)

data_source = 'MedauroideaC00' # ['CarausiusC00', 'AretaonC00', 'MedauroideaC00', 'MedauroideaC00T', 'C00', 'C00T']
No1, No2, No3 = "01", "02", "03"
No13, No14, No15 = "13", "14", "15"
No25, No26, No27 = "25", "26", "27"

# # When the data source is [all]
# vel_01, direction_01, gait_01 = get_cont_data(No1)
# vel_02, direction_02, gait_02 = get_cont_data(No2)
# vel_03, direction_03, gait_03 = get_cont_data(No3)
# acc_01, acc_02, acc_03 = calculate_acceleration(vel_01), calculate_acceleration(vel_02), calculate_acceleration(vel_03)
# vel_13, direction_13, gait_13 = get_cont_data(No13)
# vel_14, direction_14, gait_14 = get_cont_data(No14)
# vel_15, direction_15, gait_15 = get_cont_data(No15)
# acc_13, acc_14, acc_15 = calculate_acceleration(vel_13), calculate_acceleration(vel_14), calculate_acceleration(vel_15)
# if data_source == 'C00T':
#     vel_25, direction_25, gait_25 = get_cont_data(No25, trim=True, trim_len=800)
#     vel_26, direction_26, gait_26 = get_cont_data(No26, trim=True, trim_len=2200)
#     vel_27, direction_27, gait_27 = get_cont_data(No27, trim=True, trim_len=1600)
#     acc_25, acc_26, acc_27 = calculate_acceleration(vel_25), calculate_acceleration(vel_26), calculate_acceleration(vel_27)
# else:
#     vel_25, direction_25, gait_25 = get_cont_data(No25)
#     vel_26, direction_26, gait_26 = get_cont_data(No26)
#     vel_27, direction_27, gait_27 = get_cont_data(No27)
#     acc_25, acc_26, acc_27 = calculate_acceleration(vel_25), calculate_acceleration(vel_26), calculate_acceleration(vel_27)
# vel = np.concatenate((vel_01[1:], vel_02[1:], vel_03[1:], vel_13[1:], vel_14[1:], vel_15[1:], vel_25[1:], vel_26[1:], vel_27[1:]), axis=0)
# direction = np.concatenate((direction_01[1:], direction_02[1:], direction_03[1:], direction_13[1:], direction_14[1:], direction_15[1:], direction_25[1:], direction_26[1:], direction_27[1:]), axis=0)
# gait = np.concatenate((gait_01[1:], gait_02[1:], gait_03[1:], gait_13[1:], gait_14[1:], gait_15[1:], gait_25[1:], gait_26[1:], gait_27[1:]), axis=0)
# acc = np.concatenate((acc_01, acc_02, acc_03, acc_13, acc_14, acc_15, acc_25, acc_26, acc_27), axis=0)
# print("flatten trajectory length: ", len(acc))

# # When the data source is [one insect]
if data_source == 'MedauroideaC00T':
    vel_01, direction_01, gait_01 = get_cont_data(No1, trim=True, trim_len=800)
    vel_02, direction_02, gait_02 = get_cont_data(No2, trim=True, trim_len=2200)
    vel_03, direction_03, gait_03 = get_cont_data(No3, trim=True, trim_len=1600)
    acc_01, acc_02, acc_03 = calculate_acceleration(vel_01), calculate_acceleration(vel_02), calculate_acceleration(vel_03)
else:
    vel_01, direction_01, gait_01 = get_cont_data(No1)
    vel_02, direction_02, gait_02 = get_cont_data(No2)
    vel_03, direction_03, gait_03 = get_cont_data(No3)
    acc_01, acc_02, acc_03 = calculate_acceleration(vel_01), calculate_acceleration(vel_02), calculate_acceleration(vel_03)
vel = np.concatenate((vel_01[1:], vel_02[1:], vel_03[1:]), axis=0)
direction = np.concatenate((direction_01[1:], direction_02[1:], direction_03[1:]), axis=0)
gait = np.concatenate((gait_01[1:], gait_02[1:], gait_03[1:]), axis=0)
acc = np.concatenate((acc_01, acc_02, acc_03), axis=0)
print("length of T"+No1+", T"+No2+", T"+No3+": ", len(acc_01), len(acc_02), len(acc_03))
print("length of faltten trajectory:", len(acc))

# save vel and acc
plot_histogram(acc, title='Acceleration Data Distribution', xlabel='Acceleration', savename=data_source+'_histogram_acc')
plot_histogram(vel, title='Velocity Data Distribution', xlabel='Velocity', savename=data_source+'_histogram_vel')

# bin the data
vel_start, vel_end, vel_step = irl_config[data_source]['vel_bin_params']
direction_start, direction_end, direction_step = irl_config[data_source]['direction_bin_params']
acc_start, acc_end, acc_step = irl_config[data_source]['acc_bin_params']
vel_bin_edges = np.arange(vel_start, vel_end, vel_step) # the end value should be 1 unit larger
vel_binned = np.digitize(vel, vel_bin_edges, right=True)
direction_bin_edges = np.arange(direction_start, direction_end, direction_step)
direction_binned = np.digitize(direction, direction_bin_edges, right=True)
acc_bin_edges = np.arange(acc_start, acc_end, acc_step)
acc_binned = np.digitize(acc, acc_bin_edges, right=True)
# print binned group
direction_bin_group = np.unique(direction_binned)
vel_bin_group = np.unique(vel_binned)
acc_bin_group = np.unique(acc_binned)
print("direction binned group: ", direction_bin_group)
print("vel binned group: ", vel_bin_group)
print("acc binned group: ", acc_bin_group)

# Define grouped gait combinations (6 types)
grouped_gait_combinations = {
    # representative noncanonical
    '111111': 5,
    '111110': 5,
    '111101': 5,
    '111011': 5,
    '110111': 5,
    '101111': 5,
    '011111': 5,
    # tetrapod gait
    '110101': 4,
    '110011': 4,
    '101110': 4,
    '101011': 4,
    '011110': 4,
    '011101': 4,
    # tripod gait
    '101010': 3,
    '010101': 3,
    # tetrapod gait (noncanonical)
    '111010': 2,
    '111001': 2,
    '110110': 2,
    '101101': 2,
    '100111': 2, 
    '011011': 2,
    '010111': 2,
    '001111': 2, 
    # tripod gait (noncanonical)
    '110010': 1,
    '101001': 1, 
    '011010': 1,
    '011001': 1,
    '010011': 1,
    '001011': 1, 
    # rare noncanonical
    '101000': 0, 
    '100010': 0,
    '001010': 0,
    '000101': 0,
    '000010': 0,
}

# Combine the first six columns into a string for each row to represent the gait pattern
gait_data = pd.DataFrame(gait)
gait_data['Gait Pattern'] = gait_data.iloc[:, :6].astype(str).agg(''.join, axis=1)
# Categorize each gait pattern based on the possible combinations
gait_data['Category'] = gait_data['Gait Pattern'].map(grouped_gait_combinations).astype(int)
# Display the categorized data
print(gait_data[['Gait Pattern', 'Category']])

# Combine velocity, direction, and gait pattern into a single DataFrame for analysis
analysis_df = pd.DataFrame({
        'Velocity Bin': vel_binned.flatten(),
        'Acceleration Bin': acc_binned.flatten(),
        'Direction Bin': direction_binned.flatten(),
        'Gait Category': gait_data['Category']
    })

save = True
if save:
    save_path = 'expert_demonstration/expert/'+data_source+'.csv'
    analysis_df.to_csv(save_path, index=False, header=True)

# # plot states
# plot_states(vel_01, vel_02, vel_03, direction_01, direction_02, direction_03, acc_01, acc_02, acc_03, data_source)

# heat map
# heatmap_direction_vel_reward(analysis_df)
# heatmap_direction_vel_action(vel_binned, direction_binned)