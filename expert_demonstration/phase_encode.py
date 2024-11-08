import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

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

def Kalman1D(observations,damping=1):
    # to return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.03
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def smooth(data):
    smoothed_data = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        smoothed_data[:, i] = Kalman1D(data[:, i], damping=1).reshape(-1)
    return smoothed_data

def time_eplased_antenna_contact(joint_data):
# This funtion calculates the time elapsed since the last antenna contact
# i.e. the time since the last valley in the joint data
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

def antenna_visualization(original_data, clustered_data, lable, save=False):
    # subplot the encoded antenna data and the original antenna data
    plt.figure(figsize=(20, 10))  # Adjusted to better fit two y-axes for each subplot
    # HS left
    plt.subplot(4, 1, 1)
    plt.plot(original_data[:, 0], label='Original')
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # Create a second y-axis for the encoded data
    ax2.step(range(len(clustered_data)), clustered_data[:,0], where='post', color='orange', label=lable, linestyle='-')
    ax1.set_ylabel('rad')
    ax2.set_ylabel('time elapsed')
    plt.title("HS left")
    # HS right
    plt.subplot(4, 1, 2)
    plt.plot(original_data[:, 1], label='Original')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.step(range(len(clustered_data)), clustered_data[:,1], where='post', color='orange', label=lable, linestyle='-')
    ax1.set_ylabel('rad')
    ax2.set_ylabel('time elapsed')
    plt.title("HS right")
    # SP left
    plt.subplot(4, 1, 3)
    plt.plot(original_data[:, 2], label='Original')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.step(range(len(clustered_data)), clustered_data[:,2], where='post', color='orange', label=lable, linestyle='-')
    ax1.set_ylabel('rad')
    ax2.set_ylabel('time elapsed')
    plt.title("SP left")
    # SP right
    plt.subplot(4, 1, 4)
    plt.plot(original_data[:, 3], label='Original')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.step(range(len(clustered_data)), clustered_data[:,3], where='post', color='orange', label=lable, linestyle='-')
    ax1.set_ylabel('rad')
    ax2.set_ylabel('time elapsed')
    plt.title("SP right")
    plt.tight_layout()
    if save:
        plt.savefig(f"{lable}.png")
    else:
        plt.show()

antenna_01 = get_data("01")
smoothed_antenna_01 = smooth(antenna_01)
encoded_antenna_01 = time_eplased_antenna_contact(smoothed_antenna_01)
np.savetxt("encoded_antenna_01.csv", encoded_antenna_01, delimiter=",")

# path = "antenna_01.csv"
# encoded_antenna_01 = pd.read_csv(path, header=None).to_numpy()
# discretize the data: binning by log scale
log_transformed_data = np.log1p(encoded_antenna_01) 
# KMeans clustering
kmeans_log = KMeans(n_clusters=10)
discrete_data = kmeans_log.fit_predict(log_transformed_data.flatten().reshape(-1, 1))
discrete_data = discrete_data.reshape(log_transformed_data.shape)
# Save the discretized data

# visualize the encoded antenna data and the original antenna data
antenna_visualization(antenna_01, smoothed_antenna_01, 'smoothed', save=True)
antenna_visualization(antenna_01, encoded_antenna_01, 'time elapsed', save=True)
# antenna_visualization(antenna_01, log_transformed_data, 'log time elapsed', save=True)
# antenna_visualization(antenna_01, discrete_data, 'discrete time elapsed', save=False)

