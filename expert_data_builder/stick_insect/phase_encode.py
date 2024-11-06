import numpy as np
import pandas as pd
import json
import os
from scipy.signal import find_peaks

def time_since_last_valley(joint_data, threshold_intervals):
    inverted_data = (-joint_data)
    valleys, _ = find_peaks(inverted_data)
    # Initialize an array to store the time since last valley
    time_since_valley = np.zeros_like(joint_data, dtype=float)
    # Loop through the joint data to calculate the time since the last valley
    last_valley = 0
    for i in range(1, len(joint_data)):
        # Check if the current point is a valley
        if i in valleys:
            last_valley = i
        # Calculate time since the last valley
        time_since_valley[i] = i - last_valley
    return time_since_valley

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
encoded_antenna_01 = time_since_last_valley(antenna_01, 0)

# plot
import matplotlib.pyplot as plt
plt.plot(encoded_antenna_01)
plt.show()
