import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter

def Kalman1D(observations,damping=1):
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

def data_smooth(data):
    for i in range(data.shape[1]):
        smoothed_data = Kalman1D(data[:,i], damping=1).reshape(-1,1)
        data[:,i] = smoothed_data[:,0]
    return data

def peak_detection(data):
    # data: 1D array
    peak_indices = []
    for i in range(1, len(data)-1):
        if data[i] > data[i-1] and data[i] >= data[i+1]:
            peak_indices.append(i)
    return peak_indices

def valley_detection(data):
    # data: 1D array
    valley_indices = []
    for i in range(1, len(data)-1):
        if data[i] < data[i-1] and data[i] <= data[i+1]:
            valley_indices.append(i)
    return valley_indices

def plot_gait_phase(data, reverse=False):
    peak_indices = peak_detection(data)
    valley_indices = valley_detection(data)
    if peak_indices[0] < valley_indices[0]:
        # add the first data point as a valley
        valley_indices.insert(0, 0)
    else:
        # add the first data point as a peak
        peak_indices.insert(0, 0)
    if peak_indices[-1] < valley_indices[-1]:
        # add the last data point as a peak
        peak_indices.append(len(data)-1)
    else:
        # add the last data point as a valley
        valley_indices.append(len(data)-1)
    # left legs or right legs
    if not reverse:
        pass
    elif reverse:
        temp = peak_indices
        peak_indices = valley_indices
        valley_indices = temp
    # plot the data
    t = np.arange(len(data))
    plt.plot(t, data)
    # draw vertical lines at peak indices to show the gait cycle
    for i in peak_indices:
        plt.axvline(x=i, color='grey')
    # stance phase: peak to valley, green
    # swing phase: valley to peak, orange
    if peak_indices[0] < valley_indices[0]:
        for i in range(len(peak_indices)-1):
            plt.axvspan(peak_indices[i], valley_indices[i], facecolor='g', alpha=0.3)
            plt.axvspan(valley_indices[i], peak_indices[i+1], facecolor='orange', alpha=0.3)
        if peak_indices[-1] < valley_indices[-1]:
            plt.axvspan(peak_indices[-1], valley_indices[-1], facecolor='g', alpha=0.3)
    else:
        for i in range(len(valley_indices)-1):
            plt.axvspan(valley_indices[i], peak_indices[i], facecolor='orange', alpha=0.3)
            plt.axvspan(peak_indices[i], valley_indices[i+1], facecolor='g', alpha=0.3)
        if peak_indices[-1] > valley_indices[-1]:
            plt.axvspan(valley_indices[-1], peak_indices[-1], facecolor='orange', alpha=0.3)
    # plt.xlabel('Frame')
    # plt.ylabel('Joint Angle')
    # plt.title('Gait Phase')
    #plt.show()

# temperary test for c21-0680 data
fold_path = os.getcwd() + '/expert_data_builder'
cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join(fold_path, 'joint_movement', cricket_number, 
                          f'PIC{video_number}_Joint_movement.csv')
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0])
joint_movement = joint_movement.values
joint_movement = data_smooth(joint_movement)
# plot_gait_phase(joint_movement[:,1])
# plt.show()

# subplot for ThC joints
joint_movement = joint_movement[100:400,:]
ylabel = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
reverse_list_ThC = [False, False, False, True, True, True]
plt.figure(figsize=(10,8))
for i in range(6):
    plt.subplot(6,1,i+1)
    plot_gait_phase(joint_movement[:,i], reverse=reverse_list_ThC[i])
    plt.ylabel(ylabel[i])
    plt.xticks([])
plt.xlabel('Frame')
plt.suptitle('Gait Phase of ThC Joints')
plt.tight_layout()
plt.subplots_adjust(hspace=0.07)
plt.show()

# subplot for FTi joints
plt.figure(figsize=(10,8))
reverse_list_FTi = [True, False, True, False, True, False]
for i in range(6):
    plt.subplot(6,1,i+1)
    plot_gait_phase(joint_movement[:,i+6], reverse=reverse_list_FTi[i])
    plt.ylabel(ylabel[i])
    plt.xticks([])
plt.xlabel('Frame')
plt.suptitle('Gait Phase of FTi Joints')
plt.tight_layout()
plt.subplots_adjust(hspace=0.07)
plt.show()