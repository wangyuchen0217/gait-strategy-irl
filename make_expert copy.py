import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py    
import yaml
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter

# normalization
def data_scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled

def normalize(data):
    for i in range(data.shape[1]):
        data[:,i] = data_scale(data[:,i].reshape(-1,1)).reshape(-1)
    return data

# smooth the data
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

'''firl-3d  ThC joint smoothed data position'''
cricket_number = 'c21'
video_number = '0680'
joint_path = os.path.join("expert_data_builder/movement", cricket_number, 
                                                f"PIC{video_number}_Joint_movement.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=[0]).to_numpy()
smoothed_data = data_smooth(joint_movement) # smooth the data
# plot the data and smoothed data using subplot
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(joint_movement[:500,0])
plt.title('Original data')
plt.subplot(2, 1, 2)
plt.plot(smoothed_data[:500,0])
plt.title('Smoothed data')
plt.savefig('ThC_joint_smoothed_data.png')
