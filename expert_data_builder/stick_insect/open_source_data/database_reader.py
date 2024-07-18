import scipy
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def mat_reader(subject:str, visualizaiton=False):
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

    LF_fem, LM_fem, LH_fem = LF['fem'][0, 0], LM['fem'][0, 0], LH['fem'][0, 0]
    RF_fem, RM_fem, RH_fem = RF['fem'][0, 0], RM['fem'][0, 0], RH['fem'][0, 0]
    LF_fem, LM_fem, LH_fem = LF_fem['angle'][0, 0].reshape(-1,1 ), LM_fem['angle'][0, 0].reshape(-1,1 ), LH_fem['angle'][0, 0].reshape(-1,1 )
    RF_fem, RM_fem, RH_fem = RF_fem['angle'][0, 0].reshape(-1,1 ), RM_fem['angle'][0, 0].reshape(-1,1 ), RH_fem['angle'][0, 0].reshape(-1,1 )

    LF_tib, LM_tib, LH_tib = LF['tib'][0, 0], LM['tib'][0, 0], LH['tib'][0, 0]
    RF_tib, RM_tib, RH_tib = RF['tib'][0, 0], RM['tib'][0, 0], RH['tib'][0, 0]
    LF_tib, LM_tib, LH_tib = LF_tib['angle'][0, 0].reshape(-1,1 ), LM_tib['angle'][0, 0].reshape(-1,1 ), LH_tib['angle'][0, 0].reshape(-1,1 )
    RF_tib, RM_tib, RH_tib = RF_tib['angle'][0, 0].reshape(-1,1 ), RM_tib['angle'][0, 0].reshape(-1,1 ), RH_tib['angle'][0, 0].reshape(-1,1 )

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
    save_path = 'expert_data_builder/stick_insect/Carausius/' + file_name + '.csv'
    dataset.to_csv(save_path, index=False)
    print(dataset.shape)

    # write the dataset length to trail_details.json
    trail_details[f"T{subject}"]["length"] = dataset.shape[0]
    with open("configs/trail_details.json", "w") as f:
        json.dump(trail_details, f, indent=4)

    if visualizaiton:
        plt.figure(figsize=(8, 6))
        plt.plot(LF_sup, label='Protraction/Retraction', c="green")
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
    subjects = 12
    for i in range(1, subjects + 1):
        subject_number = f"{i:02}"
        mat_reader(subject_number, visualizaiton=False)