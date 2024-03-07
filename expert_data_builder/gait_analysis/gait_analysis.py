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
            
def gait_generate(data, reverse=False):
    # detect peak and valley
    peak_indices = peak_detection(data)
    valley_indices = valley_detection(data)
    # add the first data point as peak or valley
    if peak_indices[0] < valley_indices[0]:
        valley_indices.insert(0, 0)
    else:
        peak_indices.insert(0, 0)
    # add the last data point as peak or valley
    if peak_indices[-1] < valley_indices[-1]:
        peak_indices.append(len(data)-1)
    else:
        valley_indices.append(len(data)-1)
    # check if reverse is needed
    if not reverse:
        pass
    elif reverse:
        temp = peak_indices
        peak_indices = valley_indices
        valley_indices = temp
    # generate gait phase 
    # if it is stance, append 1, if it is swing, append 0
    gait_phase = []
    # begin with peak (end with peak)
    if peak_indices[0] < valley_indices[0]:
        for i in range(len(peak_indices)-1):
            gait_phase.extend([1]*(valley_indices[i]-peak_indices[i]))
            gait_phase.extend([0]*(peak_indices[i+1]-valley_indices[i]))
        # end with valley
        if peak_indices[-1] < valley_indices[-1]:
            gait_phase.extend([1]*(valley_indices[-1]-peak_indices[-1]))
    else: # begin with valley (end with valley)
        for i in range(len(valley_indices)-1):
            gait_phase.extend([0]*(peak_indices[i]-valley_indices[i]))
            gait_phase.extend([1]*(valley_indices[i+1]-peak_indices[i]))
        # end with peak
        if valley_indices[-1] < peak_indices[-1]:
            gait_phase.extend([0]*(peak_indices[-1]-valley_indices[-1]))
    return gait_phase

def plot_gait_phase(data, reverse=False):
    # detect peak and valley
    peak_indices = peak_detection(data)
    valley_indices = valley_detection(data)
    # add the first data point as peak or valley
    if peak_indices[0] < valley_indices[0]:
        valley_indices.insert(0, 0)
    else:
        peak_indices.insert(0, 0)
    # add the last data point as peak or valley
    if peak_indices[-1] < valley_indices[-1]:
        peak_indices.append(len(data)-1)
    else:
        valley_indices.append(len(data)-1)
    # check if reverse is needed
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
    # begin with peak (end with peak)
    if peak_indices[0] < valley_indices[0]:
        for i in range(len(peak_indices)-1):
            plt.axvspan(peak_indices[i], valley_indices[i], facecolor='g', alpha=0.3)
            plt.axvspan(valley_indices[i], peak_indices[i+1], facecolor='orange', alpha=0.3)
        # end with valley
        if peak_indices[-1] < valley_indices[-1]:
            plt.axvspan(peak_indices[-1], valley_indices[-1], facecolor='g', alpha=0.3)
    else: # begin with valley (end with valley)
        for i in range(len(valley_indices)-1):
            plt.axvspan(valley_indices[i], peak_indices[i], facecolor='orange', alpha=0.3)
            plt.axvspan(peak_indices[i], valley_indices[i+1], facecolor='g', alpha=0.3)
        # end with peak
        if valley_indices[-1] < peak_indices[-1]:
            plt.axvspan(valley_indices[-1], peak_indices[-1], facecolor='orange', alpha=0.3)
    # plt.xlabel('Frame')
    # plt.ylabel('Joint Angle')
    # plt.title('Gait Phase')
    #plt.show()

# temperary test for c21-0680 data
fold_path = os.getcwd() + '/expert_data_builder'
cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join(fold_path, 'movement', cricket_number, 
                          f'PIC{video_number}_Joint_movement.csv')
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0])
joint_movement = joint_movement.values
joint_movement = data_smooth(joint_movement)

# generate gait phase
joint_movement = joint_movement[:40,:]
reverse_list_ThC = [False, False, False, True, True, True]
reverse_list_FTi = [True, False, True, False, True, False]
gait_phase_ThC = np.zeros((len(joint_movement)-1,6))
gait_phase_FTi = np.zeros((len(joint_movement)-1,6))
for i in range(6):
    gait_phase_ThC[:,i] = gait_generate(joint_movement[:,i], reverse=reverse_list_ThC[i])
    gait_phase_FTi[:,i] = gait_generate(joint_movement[:,i+6], reverse=reverse_list_FTi[i])
gait_phase = np.concatenate((gait_phase_ThC, gait_phase_FTi), axis=1)
save_path = os.path.join(fold_path, 'gait_analysis', f'PIC{video_number}_gait_phase.csv')
pd.DataFrame(gait_phase).to_csv(save_path, 
                                header=["ThC_LF","ThC_LM","ThC_LH","ThC_RF","ThC_RM","ThC_RH", 
                                        "FTi_LF","FTi_LM","FTi_LH","FTi_RF","FTi_RM", "FTi_RH"], index=None)

# subplot for ThC joints
joint_movement = joint_movement[:40,:]
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
plt.savefig('expert_data_builder/gait_analysis/gait_phase_ThC.png')

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
plt.savefig('expert_data_builder/gait_analysis/gait_phase_FTi.png')