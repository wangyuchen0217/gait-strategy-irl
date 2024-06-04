# plot the data from expert_data_builder/movement/c21/PIC0680_Heading_direction.csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

'''joint movement & forces'''

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

# read the data
joint_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22.csv'
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
forces_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22_forces.csv'
forces = pd.read_csv(forces_path, header=[0], index_col=None).to_numpy()
forces_unsmoothed = forces.copy()
forces = data_smooth(forces)

# subplot
fig, axs = plt.subplots(3, 1, figsize=(15, 15))
axs[0].plot(joint_movement[:,0])
axs[0].plot(joint_movement[:,6])
axs[0].plot(joint_movement[:,12])
axs[0].plot(joint_movement[:,18])
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Joint Movement')
axs[0].set_title('Carausius_110415_00_22_joint_movement')
axs[0].grid()

axs[1].plot(forces[:,0])
axs[1].plot(forces[:,6])
axs[1].plot(forces[:,12])
axs[1].plot(forces[:,18])
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Forces')
axs[1].set_title('Carausius_110415_00_22_forces')
axs[1].grid()

axs[2].plot(forces_unsmoothed[:,0])
axs[2].plot(forces_unsmoothed[:,6])
axs[2].plot(forces_unsmoothed[:,12])
axs[2].plot(forces_unsmoothed[:,18])
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Forces_smooth')
axs[2].set_title('Carausius_110415_00_22_forces_smooth')
axs[2].grid()
plt.show()

'''direction'''

# read the data
direction_path = 'expert_data_builder/movement/c21/PIC0680_Heading_direction.csv'
direction = pd.read_csv(direction_path, index_col=0, header=0).to_numpy()

# set the initial direction to 0
direction = direction - direction[0]

# plot the direction
plt.figure(figsize=(15, 5))
plt.plot(direction)
plt.xlabel('Frame')
plt.ylabel('Direction')   
plt.title('c21_0680_direction')
plt.grid()
# plt.savefig('c21_0680_direction.png')

'''trajectory'''

# calculate the trajectory
direction = direction.flatten()
direction_rad = direction * np.pi / 180
direction_x = np.cos(direction_rad)
direction_y = - np.sin(direction_rad)
x, y = 0, 0
trajectory_x = []
trajectory_y = []
trajectory_x.append(x)
trajectory_y.append(y)
for i in range(len(direction)):
    x += 1 * direction_x[i]
    y += 1 * direction_y[i]
    trajectory_x.append(x)
    trajectory_y.append(y)

trajectory_x = np.array(trajectory_x)
trajectory_y = np.array(trajectory_y)
print("trajectory_x: ", trajectory_x.shape)
print("trajectory_y: ", trajectory_y.shape)

# plot the trajectory
plt.figure()
plt.plot(trajectory_x, trajectory_y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('c21_0680_trajectory')
plt.grid()
# plt.savefig('c21_0680_trajectory.png')