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

def smooth(data, damping=1):
    smoothed_data = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        smoothed_data[:, i] = Kalman1D(data[:, i], damping=damping).reshape(-1)
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

def save_discrete_data(subject, discrete_data):
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)
        insect_name = trail_details[f"T{subject}"]["insect_name"]
        insect_number = trail_details[f"T{subject}"]["insect_number"]
        id_1 = trail_details[f"T{subject}"]["id_1"]
        id_2 = trail_details[f"T{subject}"]["id_2"]
        id_3 = trail_details[f"T{subject}"]["id_3"]
    discrete_data = pd.DataFrame(discrete_data, columns=['HS_left', 'HS_right', 'SP_left', 'SP_right'], index=None)
    save_path = os.path.join("expert_data_builder/stick_insect", insect_name,
                                                        f"{insect_number}_{id_1}_{id_2}_{id_3}_antenna_dist.csv")
    discrete_data.to_csv(save_path, index=False, header=True)

def antenna_visualization(original_data, clustered_data, label, save=False, fontsize=16, subject="01"):
    titles = ["HS left", "HS right", "SP left", "SP right"]
    plt.figure(figsize=(12, 10))
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(original_data[:, i], color='#394E86', linewidth='2', label='Original')
        ax1 = plt.gca()
        ax2 = ax1.twinx()  # Create a second y-axis for the encoded data
        ax2.step(range(len(clustered_data)), clustered_data[:, i], where='post', color='red', linewidth='2', label=label, linestyle='--')
        # Set labels for both y-axes
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_yticks(ax1.get_yticks())
        ax1.tick_params(axis='both', labelsize=14)
        ax1.set_ylabel('rad', fontsize=fontsize)
        ax2.set_ylabel('time elapsed', fontsize=fontsize)
        plt.title(titles[i], fontsize=fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(f"expert_demonstration/expert/plot/Carausius_T{subject}_antenna_{label}.png")
    else:
        plt.show()

def plot_time_elapsed_histogram_subplots(data, bin_step, label, save=False, subject="01"):
    column_names=['HS left', 'HS right', 'SP left', 'SP right']
    data = pd.DataFrame(data, columns=column_names)
    num_columns = len(column_names)
    fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(20, 5), sharey=True)
    for i, col in enumerate(column_names):
        count_per_bin = data[col].value_counts().sort_index()
        axes[i].bar(count_per_bin.index, count_per_bin.values, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution for {col}')
        x_ticks = count_per_bin.index
        axes[i].set_xticks(x_ticks)
        axes[i].set_xticklabels([int(tick * bin_step) for tick in x_ticks])
        axes[i].set_xlabel('Time Elapsed (t)')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5) 
    # Set common y-axis label
    fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical')
    plt.suptitle('Distribution of Discrete Antenna Time Elapsed Bins')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    if save:
        plt.savefig(f"expert_demonstration/expert/plot/Carausius_T{subject}_antenna_histogram_{label}.png")
    else:
        plt.show()

def get_antenna_dist(subject:str, bin_step=60):
    # Load the antenna data
    antenna_01 = get_data(subject)
    # Smooth the antenna data: detect the contact points
    smoothed_antenna_01 = smooth(antenna_01, damping=2)
    # Calculate the time elapsed since the last antenna contact (valley)
    t_elps_antenna_01 = time_eplased_antenna_contact(smoothed_antenna_01)

    # Discretize the data: binning
    min_val = np.min(t_elps_antenna_01)
    max_val = np.max(t_elps_antenna_01)
    print(f"min_val: {min_val}, max_val: {max_val}")
    bin_edges = np.arange(min_val, max_val+bin_step, bin_step)
    discrete_data = np.digitize(t_elps_antenna_01, bin_edges)

    # # Save the discrete data
    # save_discrete_data(subject, discrete_data)

    # Visualize the encoded antenna data and the original antenna data
    antenna_visualization(antenna_01, smoothed_antenna_01, 'smoothed_dp2', subject=subject, save=True)
    antenna_visualization(antenna_01, t_elps_antenna_01, 'time_elapsed_dp2', subject=subject, save=True)
    antenna_visualization(antenna_01, discrete_data, 'discretized_dp2', subject=subject, save=True)
    plot_time_elapsed_histogram_subplots(discrete_data, bin_step, 'dp2', subject=subject, save=True)

    return discrete_data

if __name__ == "__main__":
    subject= "03"
    # Load the antenna data
    antenna_01 = get_data(subject)
    # # Smooth the antenna data: detect the contact points
    # smoothed_antenna_01 = smooth(antenna_01, damping=3)
    # Calculate the time elapsed since the last antenna contact (valley)
    t_elps_antenna_01 = time_eplased_antenna_contact(antenna_01)

    # Discretize the data: binning
    min_val = np.min(t_elps_antenna_01)
    max_val = np.max(t_elps_antenna_01)
    print(f"min_val: {min_val}, max_val: {max_val}")
    bin_step = 60
    bin_edges = np.arange(min_val, max_val+bin_step, bin_step)
    discrete_data = np.digitize(t_elps_antenna_01, bin_edges)

    # Save the discrete data
    save_discrete_data(subject, discrete_data)

    # Visualize the encoded antenna data and the original antenna data
    # antenna_visualization(antenna_01, smoothed_antenna_01, 'smoothed', subject=subject, save=True)
    antenna_visualization(antenna_01, t_elps_antenna_01, 'time_elapsed_orgl', subject=subject, save=True)
    antenna_visualization(antenna_01, discrete_data, 'discretized_orgl', subject=subject, save=True)
    plot_time_elapsed_histogram_subplots(discrete_data, bin_step, subject=subject, save=True)