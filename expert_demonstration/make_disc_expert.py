import os
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
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

vel, direction, gait = get_cont_data("01")
# bin the data
vel_bin_edges = np.arange(0, 55, 5)
vel_binned = np.digitize(vel, vel_bin_edges, right=False)
direction_bin_edges = np.arange(-20, 5, 5)
direction_binned = np.digitize(direction, direction_bin_edges, right=False)


# Define all possible gait pattern combinations (42 types)
possible_combinations = {
    '111111': 'Bin 1: All 6 legs stance',
    '111110': 'Bin 2: 5 legs stance',
    '111101': 'Bin 3: 5 legs stance',
    '111011': 'Bin 4: 5 legs stance',
    '110111': 'Bin 5: 5 legs stance',
    '101111': 'Bin 6: 5 legs stance',
    '011111': 'Bin 7: 5 legs stance',
    '111100': 'Bin 8: 4 legs stance',
    '111010': 'Bin 9: 4 legs stance',
    '111001': 'Bin 10: 4 legs stance',
    '110110': 'Bin 11: 4 legs stance',
    '110101': 'Bin 12: 4 legs stance',
    '110011': 'Bin 13: 4 legs stance',
    '101110': 'Bin 14: 4 legs stance',
    '101101': 'Bin 15: 4 legs stance',
    '101011': 'Bin 16: 4 legs stance',
    '100111': 'Bin 17: 4 legs stance',
    '011110': 'Bin 18: 4 legs stance',
    '011101': 'Bin 19: 4 legs stance',
    '011011': 'Bin 20: 4 legs stance',
    '010111': 'Bin 21: 4 legs stance',
    '001111': 'Bin 22: 4 legs stance',
    '111000': 'Bin 23: 3 legs stance',
    '110100': 'Bin 24: 3 legs stance',
    '110010': 'Bin 25: 3 legs stance',
    '110001': 'Bin 26: 3 legs stance',
    '101100': 'Bin 27: 3 legs stance',
    '101010': 'Bin 28: 3 legs stance',
    '101001': 'Bin 29: 3 legs stance',
    '100110': 'Bin 30: 3 legs stance',
    '100101': 'Bin 31: 3 legs stance',
    '100011': 'Bin 32: 3 legs stance',
    '011100': 'Bin 33: 3 legs stance',
    '011010': 'Bin 34: 3 legs stance',
    '011001': 'Bin 35: 3 legs stance',
    '010110': 'Bin 36: 3 legs stance',
    '010101': 'Bin 37: 3 legs stance',
    '010011': 'Bin 38: 3 legs stance',
    '001110': 'Bin 39: 3 legs stance',
    '001101': 'Bin 40: 3 legs stance',
    '001011': 'Bin 41: 3 legs stance',
    '000111': 'Bin 42: 3 legs stance'
}


# Combine the first six columns into a string for each row to represent the gait pattern
gait_data = pd.DataFrame(gait)
gait_data['Gait Pattern'] = gait_data.iloc[:, :6].astype(str).agg(''.join, axis=1)

# Categorize each gait pattern based on the possible combinations
gait_data['Category'] = gait_data['Gait Pattern'].map(possible_combinations)

# Display the categorized data
print(gait_data[['Gait Pattern', 'Category']])

# # Save the categorized data to a new CSV file (if needed)
# output_csv_path = 'path_to_save_categorized_gait_pattern.csv'  # Replace with your desired file path
# gait.to_csv(output_csv_path, index=False)
