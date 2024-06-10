# plot the data from expert_data_builder/movement/c21/PIC0680_Heading_direction.csv
import sys
sys.path.append("./") # add the root directory to the python path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

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

'''smoothed data'''
joint_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22.csv'
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
joint_movement_unsmoothed = joint_movement.copy()
joint_movement = data_smooth(joint_movement)
# subplot
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
axs[0].plot(joint_movement_unsmoothed[:,0], label='Sup')
axs[0].plot(joint_movement_unsmoothed[:,6], label='CTr')
axs[0].plot(joint_movement_unsmoothed[:,12], label='ThC')
axs[0].plot(joint_movement_unsmoothed[:,18], label='FTi')
axs[0].set_xlabel('Frame', fontsize=12)
axs[0].set_ylabel('Joint Movement', fontsize=12)
axs[0].set_title('Carausius_110415_00_22_joint_movement', fontsize=12)
axs[0].legend()
axs[0].grid()

axs[1].plot(joint_movement[:,0])
axs[1].plot(joint_movement[:,6])
axs[1].plot(joint_movement[:,12])
axs[1].plot(joint_movement[:,18])
axs[1].set_xlabel('Frame', fontsize=12)
axs[1].set_ylabel('Joint Movement_smooth', fontsize=12)
axs[1].set_title('Carausius_110415_00_22_joint_movement_smooth', fontsize=12)
axs[1].grid()
# plt.show()
plt.savefig('Carausius_110415_00_22_joint_movement.png')

'''joint movement & forces'''

# read the data
joint_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22.csv'
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
forces_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22_forces.csv'
forces = pd.read_csv(forces_path, header=[0], index_col=None).to_numpy()
forces_unsmoothed = forces.copy()
forces = data_smooth(forces)
leg_lengths = np.array([0.13, 0.14, 0.15, 0.13, 0.14, 0.15, 
                        0.13, 0.14, 0.15, 0.13, 0.14, 0.15, 
                        1.58, 1.16, 1.39, 1.58, 1.16, 1.39, 
                        1.5, 1.12, 1.41, 1.5, 1.12, 1.41])
torques = np.zeros(forces.shape)
for i in range(len(forces)):
    torques[i] = forces[i] * leg_lengths

# subplot
fig, axs = plt.subplots(4, 1, figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)
axs[0].plot(joint_movement[:,0], label='Sup')
axs[0].plot(joint_movement[:,6], label='CTr')
axs[0].plot(joint_movement[:,12], label='ThC')
axs[0].plot(joint_movement[:,18], label='FTi')
axs[0].set_xlabel('Frame', fontsize=12)
axs[0].set_ylabel('Joint Movement', fontsize=12)
axs[0].set_title('Carausius_110415_00_22_joint_movement', fontsize=12)
axs[0].legend()
axs[0].grid()

axs[1].plot(forces[:,0])
axs[1].plot(forces[:,6])
axs[1].plot(forces[:,12])
axs[1].plot(forces[:,18])
axs[1].set_xlabel('Frame', fontsize=12)
axs[1].set_ylabel('Forces' , fontsize=12)
axs[1].set_title('Carausius_110415_00_22_forces', fontsize=12)
axs[1].grid()

axs[2].plot(torques[:,0])
axs[2].plot(torques[:,6])
axs[2].plot(torques[:,12])
axs[2].plot(torques[:,18])
axs[2].set_xlabel('Frame', fontsize=12)
axs[2].set_ylabel('Torques', fontsize=12)
axs[2].set_title('Carausius_110415_00_22_torques', fontsize=12)
axs[2].grid()

axs[3].plot(forces_unsmoothed[:,0])
axs[3].plot(forces_unsmoothed[:,6])
axs[3].plot(forces_unsmoothed[:,12])
axs[3].plot(forces_unsmoothed[:,18])
axs[3].set_xlabel('Frame', fontsize=12)
axs[3].set_ylabel('Forces_smooth', fontsize=12)
axs[3].set_title('Carausius_110415_00_22_forces_smooth', fontsize=12)
axs[3].grid()
# plt.show()
plt.savefig('Carausius_110415_00_22.png')

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