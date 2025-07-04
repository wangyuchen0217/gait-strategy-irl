'''
This code is used to read the mat. file of stick insect data and save the data as csv file.

The data includes:
1. Joint angle of each leg
2. Velocity
3. Direction
4. Gait pattern
5. Antenna angle

Those data will be saved in the folder e.g. 'expert_data_builder/stick_insect/Carausius/'
'''

import scipy
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def json_reader(subject:str):
    # Load the .mat file
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)
    insect_name = trail_details[f"T{subject}"]["insect_name"]
    insect_number = trail_details[f"T{subject}"]["insect_number"]
    id_1 = trail_details[f"T{subject}"]["id_1"]
    id_2 = trail_details[f"T{subject}"]["id_2"]
    id_3 = trail_details[f"T{subject}"]["id_3"]
    file_name = f"{insect_number}_{id_1}_{id_2}_{id_3}"
    mat_file_path = f"expert_data_builder/stick_insect/open_source_data/{file_name}.mat"
    mat_contents = scipy.io.loadmat(mat_file_path)

    # Inspect the keys of the dictionary to understand the structure of the file
    print(mat_contents.keys())

    return mat_contents, file_name, insect_name


def mat_reader_vel(subject:str, save=False, visualizaiton=False):
    # Load the .mat file
    mat_contents, file_name, insect_name = json_reader(subject)
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)  
    # Extract specific data from the dictionary
    gait = mat_contents['gait']
    vel = gait['velocity'][0, 0]
    vel = pd.DataFrame(vel, index=None, columns=["vel"])

    if save:
        save_path = 'expert_data_builder/stick_insect/'+ insect_name +'/' + file_name + '_vel.csv'
        vel.to_csv(save_path, index=False)
        print(vel.shape) # the 1st value is nan

    if visualizaiton:
        plt.figure(figsize=(12, 3))
        plt.plot(vel)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('Vel', fontsize=14)
        plt.grid()
        plt.show()


def mat_reader_direction(subject:str, save=False, visualizaiton=False):
    # Load the .mat file
    mat_contents, file_name, insect_name = json_reader(subject)
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)  
    # Extract specific data from the dictionary    
    T3 = mat_contents['T3']
    T3_angle = T3['angle'][0, 0]
    T3_yaw = T3_angle[:, 0].reshape(-1, 1)
    direction = pd.DataFrame(T3_yaw, index=None, columns=["direction"])
    
    if save:
        save_path = 'expert_data_builder/stick_insect/'+ insect_name +'/' + file_name + '_direction.csv'
        direction.to_csv(save_path, index=False)
        print(direction.shape)

    if visualizaiton:
        plt.figure(figsize=(12, 3))
        plt.plot(direction)
        plt.ylim(-20, 20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('Direction (deg)', fontsize=14)
        plt.grid()
        plt.show()

def mat_antenna_reader(subject:str, save=False, visualizaiton=False):
    # Load the .mat file
    mat_contents, file_name, insect_name = json_reader(subject)
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)  
    # Extract specific data from the dictionary    
    antL, antR = mat_contents['antL'], mat_contents['antR']
    scpL, scpR, pedL, pedR = antL['scp'][0, 0], antR['scp'][0, 0], antL['ped'][0, 0], antR['ped'][0, 0]
    scpL_angle, scpR_angle = scpL['angle'][0, 0].reshape(-1, 1), scpR['angle'][0, 0].reshape(-1, 1)
    pedL_angle, pedR_angle = pedL['angle'][0, 0].reshape(-1, 1), pedR['angle'][0, 0].reshape(-1, 1)
    antenna = np.concatenate((scpL_angle, scpR_angle, pedL_angle, pedR_angle), axis=1)
    antenna = pd.DataFrame(antenna, index=None, columns=["scpL", "scpR", "pedL", "pedR"])

    if save:
        save_path = 'expert_data_builder/stick_insect/'+ insect_name +'/' + file_name + '_antenna.csv'
        antenna.to_csv(save_path, index=False)
        print(antenna.shape)

    if visualizaiton:
        plt.figure(figsize=(12, 3))
        plt.plot(antenna)
        plt.ylim(-1.5, 2.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('antenna (rad)', fontsize=14)
        plt.legend(["scpL", "scpR", "pedL", "pedR"])
        plt.grid()
        plt.show()

def mat_reader_gait(subject:str, save=False):
    # Load the .mat file
    mat_contents, file_name, insect_name = json_reader(subject)
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)  
    # Extract specific data from the dictionary
    gait = mat_contents['gait']
    pattern = gait['pattern'][0, 0]
    incontact = gait['incontact'][0, 0]
    right = pattern[:, 0:3]
    left = pattern[:, 3:6]
    gait_info = np.concatenate((left, right, incontact), axis=1)
    gait = pd.DataFrame(gait_info, index=None, columns=["LF", "LM", "LH", "RF", "RM", "RH", "incontact"])

    if save:
        save_path = 'expert_data_builder/stick_insect/'+ insect_name +'/' + file_name + '_gait.csv'
        gait.to_csv(save_path, index=False)
        print(gait.shape)


def mat_reader_joint_angle(subject:str, save=False, visualizaiton=False):
    # Load the .mat file
    mat_contents, file_name, insect_name = json_reader(subject)
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)

    # Extract specific data from the dictionary
    LF, LM, LH = mat_contents['L1'], mat_contents['L2'], mat_contents['L3']
    RF, RM, RH = mat_contents['R1'], mat_contents['R2'], mat_contents['R3']

    LF_cox, LM_cox, LH_cox = LF['cox'][0, 0], LM['cox'][0, 0], LH['cox'][0, 0]
    RF_cox, RM_cox, RH_cox = RF['cox'][0, 0], RM['cox'][0, 0], RH['cox'][0, 0]
    LF_cox_z, LM_cox_z, LH_cox_z = LF_cox['angle'][0, 0][:,0].reshape(-1,1 ), LM_cox['angle'][0, 0][:,0].reshape(-1,1 ), LH_cox['angle'][0, 0][:,0].reshape(-1,1 )
    LF_cox_x, LM_cox_x, LH_cox_x = LF_cox['angle'][0, 0][:,1].reshape(-1,1 ), LM_cox['angle'][0, 0][:,1].reshape(-1,1 ), LH_cox['angle'][0, 0][:,1].reshape(-1,1 )
    LF_cox_y, LM_cox_y, LH_cox_y = LF_cox['angle'][0, 0][:,2].reshape(-1,1 ), LM_cox['angle'][0, 0][:,2].reshape(-1,1 ), LH_cox['angle'][0, 0][:,2].reshape(-1,1 )
    RF_cox_z, RM_cox_z, RH_cox_z = RF_cox['angle'][0, 0][:,0].reshape(-1,1 ), RM_cox['angle'][0, 0][:,0].reshape(-1,1 ), RH_cox['angle'][0, 0][:,0].reshape(-1,1 )
    RF_cox_x, RM_cox_x, RH_cox_x = RF_cox['angle'][0, 0][:,1].reshape(-1,1 ), RM_cox['angle'][0, 0][:,1].reshape(-1,1 ), RH_cox['angle'][0, 0][:,1].reshape(-1,1 )
    RF_cox_y, RM_cox_y, RH_cox_y = RF_cox['angle'][0, 0][:,2].reshape(-1,1 ), RM_cox['angle'][0, 0][:,2].reshape(-1,1 ), RH_cox['angle'][0, 0][:,2].reshape(-1,1 )

    LF_fem_struct, LM_fem_struct, LH_fem_struct = LF['fem'][0, 0], LM['fem'][0, 0], LH['fem'][0, 0]
    RF_fem_struct, RM_fem_struct, RH_fem_struct = RF['fem'][0, 0], RM['fem'][0, 0], RH['fem'][0, 0]
    LF_fem, LM_fem, LH_fem = LF_fem_struct['angle'][0, 0].reshape(-1,1 ), LM_fem_struct['angle'][0, 0].reshape(-1,1 ), LH_fem_struct['angle'][0, 0].reshape(-1,1 )
    RF_fem, RM_fem, RH_fem = RF_fem_struct['angle'][0, 0].reshape(-1,1 ), RM_fem_struct['angle'][0, 0].reshape(-1,1 ), RH_fem_struct['angle'][0, 0].reshape(-1,1 )

    LF_tib_struct, LM_tib_struct, LH_tib_struct = LF['tib'][0, 0], LM['tib'][0, 0], LH['tib'][0, 0]
    RF_tib_struct, RM_tib_struct, RH_tib_struct = RF['tib'][0, 0], RM['tib'][0, 0], RH['tib'][0, 0]
    LF_tib, LM_tib, LH_tib = LF_tib_struct['angle'][0, 0].reshape(-1,1 ), LM_tib_struct['angle'][0, 0].reshape(-1,1 ), LH_tib_struct['angle'][0, 0].reshape(-1,1 )
    RF_tib, RM_tib, RH_tib = RF_tib_struct['angle'][0, 0].reshape(-1,1 ), RM_tib_struct['angle'][0, 0].reshape(-1,1 ), RH_tib_struct['angle'][0, 0].reshape(-1,1 )

    # Calculate the joint angles
    # Coordinate: X from posterior to anterior; Y from right to left; Z from bottom to top
    # Supination / Pronation: hip joint rotates backward and forward (right leg right hand, left leg left hand)
    LF_sup, LM_sup, LH_sup = LF_cox_x, LM_cox_x, LH_cox_x
    RF_sup, RM_sup, RH_sup = RF_cox_x, RM_cox_x, RH_cox_x

    # Levation / Depression: CTr lift up and down
    LF_CTr, LM_CTr, LH_CTr = LF_cox_y + LF_fem, LM_cox_y + LM_fem, LH_cox_y + LH_fem
    RF_CTr, RM_CTr, RH_CTr = RF_cox_y + RF_fem, RM_cox_y + RM_fem, RH_cox_y + RH_fem

    # Protraction / Retraction: ThC swing forward and backward
    LF_ThC, LM_ThC, LH_ThC = LF_cox_z, LM_cox_z, LH_cox_z
    RF_ThC, RM_ThC, RH_ThC = RF_cox_z, RM_cox_z, RH_cox_z

    # Extension / Flexion: FTi stretch and contract
    LF_FTi, LM_FTi, LH_FTi = LF_tib, LM_tib, LH_tib
    RF_FTi, RM_FTi, RH_FTi = RF_tib, RM_tib, RH_tib

    # Set the dataset
    dataset = np.concatenate((LF_sup, LM_sup, LH_sup, RF_sup, RM_sup, RH_sup,
                                                        LF_CTr, LM_CTr, LH_CTr, RF_CTr, RM_CTr, RH_CTr,
                                                        LF_ThC, LM_ThC, LH_ThC, RF_ThC, RM_ThC, RH_ThC,
                                                        LF_FTi, LM_FTi, LH_FTi, RF_FTi, RM_FTi, RH_FTi), axis=1)
    dataset = pd.DataFrame(dataset, index=None, columns=["LF_sup", "LM_sup", "LH_sup", "RF_sup", "RM_sup", "RH_sup",
                                                                                                "LF_CTr", "LM_CTr", "LH_CTr", "RF_CTr", "RM_CTr", "RH_CTr",
                                                                                                "LF_ThC", "LM_ThC", "LH_ThC", "RF_ThC", "RM_ThC", "RH_ThC",
                                                                                                "LF_FTi", "LM_FTi", "LH_FTi", "RF_FTi", "RM_FTi", "RH_FTi"])

    
    # Extract the foot teajectory
    LF_tar, LM_tar, LH_tar = LF['tar'][0, 0], LM['tar'][0, 0], LH['tar'][0, 0]
    RF_tar, RM_tar, RH_tar = RF['tar'][0, 0], RM['tar'][0, 0], RH['tar'][0, 0]
    LF_tar, LM_tar, LH_tar = LF_tar['pos'][0, 0], LM_tar['pos'][0, 0], LH_tar['pos'][0, 0]
    RF_tar, RM_tar, RH_tar = RF_tar['pos'][0, 0], RM_tar['pos'][0, 0], RH_tar['pos'][0, 0]

    # Foot Trajectory
    LF_foot, LM_foot, LH_foot = LF_tar, LM_tar, LH_tar
    RF_foot, RM_foot, RH_foot = RF_tar, RM_tar, RH_tar
    print("LF_foot:", LF_foot.shape)

    foot_trajectory = np.concatenate((LF_foot, LM_foot, LH_foot, RF_foot, RM_foot, RH_foot), axis=1)
    foot_trajectory = pd.DataFrame(foot_trajectory, index=None, columns=["LF_x", "LF_y", "LF_z", "LF_w", 
                                                                                                    "LM_x", "LM_y", "LM_z", "LM_w",
                                                                                                    "LH_x", "LH_y", "LH_z", "LH_w",
                                                                                                    "RF_x", "RF_y", "RF_z", "RF_w",
                                                                                                    "RM_x", "RM_y", "RM_z", "RM_w",
                                                                                                    "RH_x", "RH_y", "RH_z", "RH_w"])
    
    # # plot foot trajectories
    # plt.figure(figsize=(15, 6))
    # plt.plot(LF_foot[:, 0], LF_foot[:, 2], label='LF Foot Trajectory', c="blue")
    # plt.plot(LM_foot[:, 0], LM_foot[:, 2], label='LM Foot Trajectory', c="orange")
    # plt.plot(LH_foot[:, 0], LH_foot[:, 2], label='LH Foot Trajectory', c="green")
    # plt.plot(RF_foot[:, 0], RF_foot[:, 2], label='RF Foot Trajectory', c="red")
    # plt.plot(RM_foot[:, 0], RM_foot[:, 2], label='RM Foot Trajectory', c="purple")
    # plt.plot(RH_foot[:, 0], RH_foot[:, 2], label='RH Foot Trajectory', c="brown")
    # plt.xlabel('X (mm)', fontsize=14)
    # plt.ylabel('Z (mm)', fontsize=14)
    # plt.title('Foot Trajectory', fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()


    # Extract the morphology data
    Hd, T1, T2, T3 = mat_contents['Hd'], mat_contents['T1'], mat_contents['T2'], mat_contents['T3']
    len_HD, len_T1, len_T2, len_T3 = Hd['LENGTH'][0, 0].reshape(-1, 1), T1['LENGTH'][0, 0].reshape(-1, 1), T2['LENGTH'][0, 0].reshape(-1, 1), T3['LENGTH'][0, 0].reshape(-1, 1)
    len_LF_fem, len_LM_fem, len_LH_fem = LF_fem_struct['LENGTH'][0, 0].reshape(-1, 1), LM_fem_struct['LENGTH'][0, 0].reshape(-1, 1), LH_fem_struct['LENGTH'][0, 0].reshape(-1, 1)
    len_LF_tib, len_LM_tib, len_LH_tib = LF_tib_struct['LENGTH'][0, 0].reshape(-1, 1), LM_tib_struct['LENGTH'][0, 0].reshape(-1, 1), LH_tib_struct['LENGTH'][0, 0].reshape(-1, 1)
    len_RF_fem, len_RM_fem, len_RH_fem = RF_fem_struct['LENGTH'][0, 0].reshape(-1, 1), RM_fem_struct['LENGTH'][0, 0].reshape(-1, 1), RH_fem_struct['LENGTH'][0, 0].reshape(-1, 1)
    len_RF_tib, len_RM_tib, len_RH_tib = RF_tib_struct['LENGTH'][0, 0].reshape(-1, 1), RM_tib_struct['LENGTH'][0, 0].reshape(-1, 1), RH_tib_struct['LENGTH'][0, 0].reshape(-1, 1)

    morphology = np.concatenate((len_HD, len_T1, len_T2, len_T3,
                                len_LF_fem, len_LM_fem, len_LH_fem, len_RF_fem, len_RM_fem, len_RH_fem,
                                len_LF_tib, len_LM_tib, len_LH_tib, len_RF_tib, len_RM_tib, len_RH_tib), axis=1)
    morphology = pd.DataFrame(morphology, index=None, columns=["Hd", "T1", "T2", "T3",
                                                                "LF_fem", "LM_fem", "LH_fem", "RF_fem", "RM_fem", "RH_fem",
                                                                "LF_tib", "LM_tib", "LH_tib", "RF_tib", "RM_tib", "RH_tib"])
    

    if save:
        save_path_joint = 'expert_data_builder/stick_insect/'+ insect_name +'/' + file_name + '.csv'
        dataset.to_csv(save_path_joint, index=False)
        print(dataset.shape)

        save_path_foot = 'expert_data_builder/stick_insect/'+ insect_name +'/' + file_name + '_foot.csv'
        foot_trajectory.to_csv(save_path_foot, index=False)

        save_path_morpho = 'expert_data_builder/stick_insect/'+ insect_name +'/' + file_name + '_morphology.csv'
        morphology.to_csv(save_path_morpho, index=False)

    # write the dataset length to trail_details.json
    trail_details[f"T{subject}"]["length"] = dataset.shape[0]
    with open("configs/trail_details.json", "w") as f:
        json.dump(trail_details, f, indent=4)

    if visualizaiton:
        plt.figure(figsize=(8, 6))
        plt.plot(LF_sup, label='Supination/Pronation', c="green")
        plt.plot(LF_CTr, label='Levation/Depression', c='red')
        plt.plot(LF_ThC, label='Protraction/Retraction', c='blue')
        plt.plot(LF_FTi, label='Extension/Flexion', c="magenta")
        plt.ylim(-90, 180)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('Joint Angle (deg)', fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    # input the total number of subjects
    subjects = 3 # 12
    # for i in range(1, subjects + 1):
    for i in range(1, 4):
        subject_number = f"{i:02}"
        mat_reader_joint_angle(subject_number, save=True, visualizaiton=False)
        # mat_reader_vel(subject_number, save=True, visualizaiton=True)
        # mat_reader_direction(subject_number, save=True, visualizaiton=True)
        # mat_reader_gait(subject_number, save=True)
        # mat_antenna_reader(subject_number, save=True, visualizaiton=True)