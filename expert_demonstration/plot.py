# plot the data from expert_data_builder/movement/c21/PIC0680_Heading_direction.csv
import sys
sys.path.append("./") # add the root directory to the python path
import matplotlib.pyplot as plt
import pandas as pd
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

def torque_plot(): # plot the torque data w/ x,y,z
    torques_path = '/home/yuchen/insect_walking_irl/expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22_torques.csv'
    torques = pd.read_csv(torques_path, header=None, index_col=None).to_numpy()
    torques = data_smooth(torques)

    # plot sup blue
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(torques[:,0])
    axs[0].set_xlabel('Frame', fontsize=14)
    axs[0].set_ylabel('Torque_x', fontsize=14)
    axs[0].set_title('Carausius_110415_00_22_sup', fontsize=14)
    axs[0].grid()

    axs[1].plot(torques[:,1])
    axs[1].set_xlabel('Frame', fontsize=14)
    axs[1].set_ylabel('Torque_y', fontsize=14)
    axs[1].set_title('Carausius_110415_00_22_sup', fontsize=14)
    axs[1].grid()

    axs[2].plot(torques[:,2])
    axs[2].set_xlabel('Frame', fontsize=14)
    axs[2].set_ylabel('Torque_z', fontsize=14)
    axs[2].set_title('Carausius_110415_00_22_sup', fontsize=14)
    axs[2].grid()
    plt.savefig("Carausius_110415_00_22_sup.png")

    # plot CTr orange
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(torques[:,18], c='orange')
    axs[0].set_xlabel('Frame', fontsize=14)
    axs[0].set_ylabel('Torque_x', fontsize=14)
    axs[0].set_title('Carausius_110415_00_22_CTr', fontsize=14)
    axs[0].grid()

    axs[1].plot(torques[:,19], c='orange')
    axs[1].set_xlabel('Frame', fontsize=14)
    axs[1].set_ylabel('Torque_y', fontsize=14)
    axs[1].set_title('Carausius_110415_00_22_CTr', fontsize=14)
    axs[1].grid()

    axs[2].plot(torques[:,20], c='orange')
    axs[2].set_xlabel('Frame', fontsize=14)
    axs[2].set_ylabel('Torque_z', fontsize=14)
    axs[2].set_title('Carausius_110415_00_22_CTr', fontsize=14)
    axs[2].grid()
    plt.savefig("Carausius_110415_00_22_CTr.png")

    # plot ThC green
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(torques[:,36], c='green')
    axs[0].set_xlabel('Frame', fontsize=14)
    axs[0].set_ylabel('Torque_x', fontsize=14)
    axs[0].set_title('Carausius_110415_00_22_ThC', fontsize=14)
    axs[0].grid()

    axs[1].plot(torques[:,37], c='green')
    axs[1].set_xlabel('Frame', fontsize=14)
    axs[1].set_ylabel('Torque_y', fontsize=14)
    axs[1].set_title('Carausius_110415_00_22_ThC', fontsize=14)
    axs[1].grid()

    axs[2].plot(torques[:,38], c='green')
    axs[2].set_xlabel('Frame', fontsize=14)
    axs[2].set_ylabel('Torque_z', fontsize=14)
    axs[2].set_title('Carausius_110415_00_22_ThC', fontsize=14)
    axs[2].grid()
    plt.savefig("Carausius_110415_00_22_ThC.png")

    # plot FTi red
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(torques[:,54], c='red')
    axs[0].set_xlabel('Frame', fontsize=14)
    axs[0].set_ylabel('Torque_x', fontsize=14)
    axs[0].set_title('Carausius_110415_00_22_FTi', fontsize=14)
    axs[0].grid()

    axs[1].plot(torques[:,55], c='red')
    axs[1].set_xlabel('Frame', fontsize=14)
    axs[1].set_ylabel('Torque_y', fontsize=14)
    axs[1].set_title('Carausius_110415_00_22_FTi', fontsize=14)
    axs[1].grid()

    axs[2].plot(torques[:,56], c='red')
    axs[2].set_xlabel('Frame', fontsize=14)
    axs[2].set_ylabel('Torque_z', fontsize=14)
    axs[2].set_title('Carausius_110415_00_22_FTi', fontsize=14)
    axs[2].grid()
    plt.savefig("Carausius_110415_00_22_FTi.png")

    # plot all
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(torques[:,0])
    axs[0].plot(torques[:,18])
    axs[0].plot(torques[:,36])
    axs[0].plot(torques[:,54])
    axs[0].set_xlabel('Frame', fontsize=14)
    axs[0].set_ylabel('Torque_x', fontsize=14)
    axs[0].set_title('Carausius_110415_00_22_torque_x', fontsize=14)
    axs[0].grid()
    axs[0].legend(['Sup', 'CTr', 'ThC', 'FTi'])

    axs[1].plot(torques[:,1])
    axs[1].plot(torques[:,19])
    axs[1].plot(torques[:,37])
    axs[1].plot(torques[:,55])
    axs[1].set_xlabel('Frame', fontsize=14)
    axs[1].set_ylabel('Torque_y', fontsize=14)
    axs[1].set_title('Carausius_110415_00_22_torque_y', fontsize=14)
    axs[1].grid()
    axs[1].legend(['Sup', 'CTr', 'ThC', 'FTi'])

    axs[2].plot(torques[:,2])
    axs[2].plot(torques[:,20])
    axs[2].plot(torques[:,38])
    axs[2].plot(torques[:,56])
    axs[2].set_xlabel('Frame', fontsize=14)
    axs[2].set_ylabel('Torque_z', fontsize=14)
    axs[2].set_title('Carausius_110415_00_22_torque_z', fontsize=14)
    axs[2].grid()
    axs[2].legend(['Sup', 'CTr', 'ThC', 'FTi'])
    plt.savefig("Carausius_110415_00_22_torque.png")

def torque_scalar_plot():
    # plot and compare the data
    joint_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22.csv'
    joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()

    torques_path = '/home/yuchen/insect_walking_irl/expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22_torques.csv'
    torques = pd.read_csv(torques_path, header=None, index_col=None).to_numpy()

    # convert the generalized data to scalar values
    torque_scalars = np.zeros((torques.shape[0], torques.shape[1]//3))
    for i in range(torques.shape[0]):
        for j in range(0, torques.shape[1], 3):
            idx = j//3
            torque_scalars[i,idx]= np.sqrt(torques[i,j]**2 + torques[i,j+1]**2 + torques[i,j+2]**2)

    torques_unsmoothed = torques.copy()
    torques = data_smooth(torques) # smooth the data
    torques_scalar_unsmoothed = torque_scalars.copy()
    torques_scalar = data_smooth(torque_scalars) # smooth the data

    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(joint_movement[:,0])
    axs[0].plot(joint_movement[:,6])
    axs[0].plot(joint_movement[:,12])
    axs[0].plot(joint_movement[:,18])
    axs[0].set_xlabel('Frame', fontsize=14)
    axs[0].set_ylabel('Joint Movement', fontsize=14)
    axs[0].set_title('Carausius_110415_00_22_joint_movement', fontsize=14)
    axs[0].grid()
    axs[0].legend(['Sup', 'CTr', 'ThC', 'FTi'])

    axs[1].plot(torques_scalar[:,0])
    axs[1].plot(torques_scalar[:,6])
    axs[1].plot(torques_scalar[:,12])
    axs[1].plot(torques_scalar[:,18])
    axs[1].set_xlabel('Frame', fontsize=14)
    axs[1].set_ylabel('Torques_smooth', fontsize=14)
    axs[1].set_title('Carausius_110415_00_22_torques_smooth', fontsize=14)
    axs[1].grid()

    axs[2].plot(torques_scalar_unsmoothed[:,0])
    axs[2].plot(torques_scalar_unsmoothed[:,6])
    axs[2].plot(torques_scalar_unsmoothed[:,12])
    axs[2].plot(torques_scalar_unsmoothed[:,18])
    axs[2].set_xlabel('Frame', fontsize=14)
    axs[2].set_ylabel('Torques', fontsize=14)
    axs[2].set_title('Carausius_110415_00_22_torques', fontsize=14)
    axs[2].grid()
    plt.savefig("Carausius_110415_00_22.png")

def smooth_method_plot():

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

def joint_forces_plot():
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

def direction_plot():
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

def trajectory_plot():
    direction_path = 'expert_data_builder/movement/c21/PIC0680_Heading_direction.csv'
    direction = pd.read_csv(direction_path, index_col=0, header=0).to_numpy()

    # set the initial direction to 0
    direction = direction - direction[0]

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