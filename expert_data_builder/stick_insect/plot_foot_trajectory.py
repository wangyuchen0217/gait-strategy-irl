'''
This code plots the foot trajectory of stick insects in the X-Z plane.
The plots are saved in the folder 'expert_data_builder/stick_insect/morphology/'.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the foot trajectory data
Aretaon_1_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_01_foot.csv'
Aretaon_2_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_04_foot.csv'
Areraon_3_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_05_foot.csv'

Carausius_1_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22_foot.csv'
Carausius_2_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_23_foot.csv'
Carausius_3_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_32_foot.csv'

Medauroidea_1_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_15_foot.csv'
Medauroidea_2_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_16_foot.csv'
Medauroidea_3_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_31_foot.csv'

def plot_foot_trajectory(path, title, save_name):
    df = pd.read_csv(path, header=[0], index_col=None)

    lf_x, lm_x, lh_x = df['LF_x'].values, df['LM_x'].values, df['LH_x'].values
    rf_x, rm_x, rh_x = df['RF_x'].values, df['RM_x'].values, df['RH_x'].values
    lf_z, lm_z, lh_z = df['LF_z'].values, df['LM_z'].values, df['LH_z'].values
    rf_z, rm_z, rh_z = df['RF_z'].values, df['RM_z'].values, df['RH_z'].values

    all_z = np.concatenate((lf_z, lm_z, lh_z, rf_z, rm_z, rh_z))
    z_min, z_max = np.min(all_z), np.max(all_z)

    # subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8)) # (12,4) for Aretaon, Carausius; (12,8) for Medauroidea
    axs[0, 0].plot(lf_x, lf_z, label='LF Foot Trajectory (X-Z)', alpha=0.8)
    axs[0, 0].set_xlabel('X Position (mm)')
    axs[0, 0].set_ylabel('Z Position (mm)')
    axs[0, 0].set_ylim(z_min * 1.2, z_max * 1.2)
    axs[0, 0].set_title('Left Front Foot Trajectory in X-Z Plane')
    axs[0, 0].grid(True)
    axs[0, 0].set_aspect('equal', adjustable='box')

    axs[0, 1].plot(rf_x, rf_z, label='RF Foot Trajectory (X-Z)', alpha=0.8)
    axs[0, 1].set_xlabel('X Position (mm)')
    axs[0, 1].set_ylabel('Z Position (mm)')
    axs[0, 1].set_ylim(z_min * 1.2, z_max * 1.2)
    axs[0, 1].set_title('Right Front Foot Trajectory in X-Z Plane')
    axs[0, 1].grid(True)
    axs[0, 1].set_aspect('equal', adjustable='box')

    axs[1, 0].plot(lm_x, lm_z, label='LM Foot Trajectory (X-Z)', alpha=0.8)
    axs[1, 0].set_xlabel('X Position (mm)')
    axs[1, 0].set_ylabel('Z Position (mm)')
    axs[1, 0].set_ylim(z_min * 1.2, z_max * 1.2)
    axs[1, 0].set_title('Left Middle Foot Trajectory in X-Z Plane')
    axs[1, 0].grid(True)
    axs[1, 0].set_aspect('equal', adjustable='box')

    axs[1, 1].plot(rm_x, rm_z, label='RM Foot Trajectory (X-Z)', alpha=0.8)
    axs[1, 1].set_xlabel('X Position (mm)')
    axs[1, 1].set_ylabel('Z Position (mm)')
    axs[1, 1].set_ylim(z_min * 1.2, z_max * 1.2)
    axs[1, 1].set_title('Right Middle Foot Trajectory in X-Z Plane')  
    axs[1, 1].grid(True)
    axs[1, 1].set_aspect('equal', adjustable='box')

    axs[2, 0].plot(lh_x, lh_z, label='LH Foot Trajectory (X-Z)', alpha=0.8)
    axs[2, 0].set_xlabel('X Position (mm)')
    axs[2, 0].set_ylabel('Z Position (mm)')
    axs[2, 0].set_ylim(z_min * 1.2, z_max * 1.2)
    axs[2, 0].set_title('Left Hind Foot Trajectory in X-Z Plane')
    axs[2, 0].grid(True)
    axs[2, 0].set_aspect('equal', adjustable='box')

    axs[2, 1].plot(rh_x, rh_z, label='RH Foot Trajectory (X-Z)', alpha=0.8)
    axs[2, 1].set_xlabel('X Position (mm)')
    axs[2, 1].set_ylabel('Z Position (mm)')
    axs[2, 1].set_ylim(z_min * 1.2, z_max * 1.2)
    axs[2, 1].set_title('Right Hind Foot Trajectory in X-Z Plane')
    axs[2, 1].grid(True)
    axs[2, 1].set_aspect('equal', adjustable='box')

    plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(f'expert_data_builder/stick_insect/morphology/{save_name}.png') # foot_trajectory_Aretaon_1

if __name__ == "__main__":
    # plot_foot_trajectory(Aretaon_1_path, 'Aretaon Trail 1', 'foot_trajectory_Aretaon_1')
    # plot_foot_trajectory(Aretaon_2_path, 'Aretaon Trail 2', 'foot_trajectory_Aretaon_2')
    # plot_foot_trajectory(Areraon_3_path, 'Aretaon Trail 3', 'foot_trajectory_Aretaon_3')

    # plot_foot_trajectory(Carausius_1_path, 'Carausius Trail 1', 'foot_trajectory_Carausius_1')
    # plot_foot_trajectory(Carausius_2_path, 'Carausius Trail 2', 'foot_trajectory_Carausius_2')
    # plot_foot_trajectory(Carausius_3_path, 'Carausius Trail 3', 'foot_trajectory_Carausius_3')

    plot_foot_trajectory(Medauroidea_1_path, 'Medauroidea Trail 1', 'foot_trajectory_Medauroidea_1')
    plot_foot_trajectory(Medauroidea_2_path, 'Medauroidea Trail 2', 'foot_trajectory_Medauroidea_2')
    plot_foot_trajectory(Medauroidea_3_path, 'Medauroidea Trail 3', 'foot_trajectory_Medauroidea_3')