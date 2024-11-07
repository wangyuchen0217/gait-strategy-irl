import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def time_eplased_antenna_contact(joint_data):
# This funtion calculates the time elapsed since the last antenna contact
# i.e. the time since the last valley in the joint data
    print(len(joint_data))
    print(len(joint_data[1]))
    time_elapsed = np.zeros_like(joint_data, dtype=float)
    for j in range(len(joint_data[1])):
        for i in range(len(joint_data)):
            inverted_data = -joint_data[:, j]
            valley_idx, _ = find_peaks(inverted_data)
            # Loop to calculate the time elapsed since the last valley
            last_valley = 0
            for i in range(1, len(joint_data)):
                # Check if the current point is a valley
                if i in valley_idx:
                    last_valley = i
                # Calculate time since the last valley
                time_elapsed[i, j] = i - last_valley
    return time_elapsed

def get_data(subject:str):
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)
        insect_name = trail_details[f"T{subject}"]["insect_name"]
        insect_number = trail_details[f"T{subject}"]["insect_number"]
        id_1 = trail_details[f"T{subject}"]["id_1"]
        id_2 = trail_details[f"T{subject}"]["id_2"]
        id_3 = trail_details[f"T{subject}"]["id_3"]
        antenna_path = os.path.join("expert_data_builder/stick_insect", insect_name,
                                                        f"{insect_number}_{id_1}_{id_2}_{id_3}_antenna.csv")
        antenna = pd.read_csv(antenna_path, header=[0], index_col=None).to_numpy()
    return antenna

antenna_01 = get_data("01")
encoded_antenna_01 = time_eplased_antenna_contact(antenna_01)

# plot
plt.plot(encoded_antenna_01)
plt.show()
