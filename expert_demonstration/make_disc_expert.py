import os
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter


def get_cont_data(subject:str):
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
    return vel, direction, gait

vel_01, direction_01, gait_01 = get_cont_data("01")
vel_02, direction_02, gait_02 = get_cont_data("02")
vel_03, direction_03, gait_03 = get_cont_data("03")
vel = np.concatenate((vel_01, vel_02, vel_03), axis=0)
direction = np.concatenate((direction_01, direction_02, direction_03), axis=0)
gait = np.concatenate((gait_01, gait_02, gait_03), axis=0)

# bin the data
vel_bin_edges = np.arange(0, 145, 5) # should be 145
vel_binned = np.digitize(vel, vel_bin_edges, right=True)
direction_bin_edges = np.arange(-20, 10, 5)
direction_binned = np.digitize(direction, direction_bin_edges, right=True)

# Define all possible gait pattern combinations (42 types)
possible_combinations = {
    '111111': 1,
    '111110': 2,
    '111101': 3,
    '111011': 4,
    '110111': 5,
    '101111': 6,
    '011111': 7,
    '111100': 8,
    '111010': 9,
    '111001': 10,
    '110110': 11,
    '110101': 12,
    '110011': 13,
    '101110': 14,
    '101101': 15,
    '101011': 16,
    '100111': 17,
    '011110': 18,
    '011101': 19,
    '011011': 20,
    '010111': 21,
    '001111': 22,
    '111000': 23,
    '110100': 24,
    '110010': 25,
    '110001': 26,
    '101100': 27,
    '101010': 28,
    '101001': 29,
    '100110': 30,
    '100101': 31,
    '100011': 32,
    '011100': 33,
    '011010': 34,
    '011001': 35,
    '010110': 36,
    '010101': 37,
    '010011': 38,
    '001110': 39,
    '001101': 40,
    '001011': 41,
    '000111': 42,
    '010100': 43,
    '010001': 44,
    '010000': 45,
}


# Combine the first six columns into a string for each row to represent the gait pattern
gait_data = pd.DataFrame(gait)
gait_data['Gait Pattern'] = gait_data.iloc[:, :6].astype(str).agg(''.join, axis=1)
# Categorize each gait pattern based on the possible combinations
gait_data['Category'] = gait_data['Gait Pattern'].map(possible_combinations).astype(int)
# Display the categorized data
print(gait_data[['Gait Pattern', 'Category']])

# Combine velocity, direction, and gait pattern into a single DataFrame for analysis
analysis_df = pd.DataFrame({
        'Velocity Bin': vel_binned.flatten(),
        'Direction Bin': direction_binned.flatten(),
        'Gait Category': gait_data['Category']
    })

save = False
if save:
    save_path = 'expert_demonstration/expert/CarausiusC00.csv'
    analysis_df.to_csv(save_path, index=False, header=True)

# heat map
heat_map_gait = True
if heat_map_gait:
    # Create a pivot table to count occurrences
    pivot_table = analysis_df.pivot_table(
                                        index='Velocity Bin', 
                                        columns='Direction Bin', 
                                        values='Gait Category', 
                                        aggfunc=lambda x: x.value_counts().index[0],
                                        fill_value=0)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f")  # Change 'fmt' if needed for proper formatting
    plt.title('Heat Map of Gait Patterns by Velocity and Direction')
    plt.xlabel('Direction Bin')
    plt.ylabel('Velocity Bin')
    plt.show()

heat_map_size = False
if heat_map_size:
    # Combine velocity and direction into a single DataFrame
    state_df = pd.DataFrame({
        'Velocity Bin': vel_binned.flatten(),
        'Direction Bin': direction_binned.flatten(),
    })

    # Create a pivot table to count occurrences of each state
    state_counts = state_df.pivot_table(index='Velocity Bin', columns='Direction Bin', aggfunc='size', fill_value=0)

    # Plot the heatmap of most accessed states
    plt.figure(figsize=(12, 8))
    sns.heatmap(state_counts, cmap='YlGnBu', annot=True, fmt="d")  # 'fmt="d"' for integer counts
    plt.title('Heat Map of Most Accessed States by Velocity and Direction')
    plt.xlabel('Direction Bin')
    plt.ylabel('Velocity Bin')
    plt.show()
