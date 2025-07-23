'''
This code plots all joint angles of stick insect legs.
The plots are saved in the folder 'expert_data_builder/stick_insect/plot_joint_angle/'.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Aretaon
# file_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_01.csv'
# file_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_04.csv'
# file_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_05.csv'

# # Carausius
# file_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22.csv'
# file_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_23.csv'
# file_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_32.csv'

# # Medauroidea
# file_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_15.csv'
# file_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_16.csv'
file_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_31.csv'

joint_movement = pd.read_csv(file_path, header=[0], index_col=None).to_numpy()
# skip the smooth part
# FTi joint angle minus 90 degree
joint_movement[:,-6:] = joint_movement[:,-6:] - 90
# remove the sup data
joint_movement = joint_movement[:,6:]
print("joint_movement:", joint_movement.shape)

# sim
# joint_movement = joint_movement[1371:2070, :]

# Plot the CTr joint angles [:,0:6] subplots
fig, axs = plt.subplots(6, 1, figsize=(8, 10))
labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
for i in range(6):
    axs[i].plot(joint_movement[:, i], label=labels[i],   color='palevioletred')
    axs[i].set_ylabel(labels[i], fontsize=16)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].grid()
axs[0].set_title('CTr Joint (deg)', fontsize=18)
axs[-1].set_xlabel('Time (frames)', fontsize=16)
plt.tight_layout()
plt.savefig('expert_data_builder/stick_insect/plot_joint_angle/Medauroidea3_CTr.png')

# Plot the ThC joint angles [:,6:12] subplots
fig, axs = plt.subplots(6, 1, figsize=(8, 10))
for i in range(6):
    axs[i].plot(joint_movement[:, i + 6], label=labels[i])
    axs[i].set_ylabel(labels[i], fontsize=16)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].grid()
axs[0].set_title('ThC Joint (deg)', fontsize=18)
axs[-1].set_xlabel('Time (frames)', fontsize=16)
plt.tight_layout()
plt.savefig('expert_data_builder/stick_insect/plot_joint_angle/Medauroidea3_ThC.png')

# Plot the FTi joint angles [:,12:18] subplots
fig, axs = plt.subplots(6, 1, figsize=(8, 10))
for i in range(6):
    axs[i].plot(joint_movement[:, i + 12], label=labels[i], color='green')
    axs[i].set_ylabel(labels[i], fontsize=16)
    axs[i].tick_params(axis='both', which='major', labelsize=14)
    axs[i].grid()
axs[0].set_title('FTi Joint (deg)', fontsize=18)
axs[-1].set_xlabel('Time (frames)', fontsize=16)
plt.tight_layout()
plt.savefig('expert_data_builder/stick_insect/plot_joint_angle/Medauroidea3_FTi.png')
