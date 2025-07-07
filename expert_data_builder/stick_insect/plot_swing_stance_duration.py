import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the gait data
Aretaon_1_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_01_gait.csv'
Aretaon_2_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_04_gait.csv'
Areraon_3_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_05_gait.csv'

Carausius_1_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22_gait.csv'
Carausius_2_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_23_gait.csv'
Carausius_3_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_32_gait.csv'

Medauroidea_1_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_15_gait.csv'
Medauroidea_2_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_16_gait.csv'
Medauroidea_3_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_31_gait.csv'

def count_stance(data):
    count = 0
    for i in range(len(data)):
        if data[i] == 1:
            count += 1
    return count   

def count_stance_swing_per_cycle(data):
    data = np.array(data)
    stance_durations = []
    swing_durations = []

    # stance starts when the data changes from 0 to 1
    stance_starts = np.where((data[:-1] == 0) & (data[1:] == 1))[0] + 1

    # a cycle: stance + swing
    for i in range(len(stance_starts) - 1):
        start = stance_starts[i]
        end = stance_starts[i + 1]

        cycle = data[start:end]
        stance_duration = np.sum(cycle == 1)
        swing_duration = np.sum(cycle == 0)

        stance_durations.append(stance_duration)
        swing_durations.append(swing_duration)

    return stance_durations, swing_durations
    



def plot_swing_stance_duration(path, title, save_name):
    front_stance, middle_stance, hind_stance = [], [], []
    front_swing, middle_swing, hind_swing = [], [], []
    for i in range(3):
        # Load the gait data for each stick insect species
        df = pd.read_csv(path[i], header=[0], index_col=None)
        
        # Trim for Medauroidea
        if title == 'Medauroidea':
            if i == 0:
                df = df.iloc[0:-800, :]
            elif i == 1:
                df = df.iloc[0:-2200, :]
            else:
                df = df.iloc[0:-1600, :]

        LF, RF = df['LF'].values, df['RF'].values
        LM, RM = df['LM'].values, df['RM'].values
        LH, RH = df['LH'].values, df['RH'].values
        # Count the stance and swing durations for each leg
        LF_stance, LF_swing = count_stance_swing_per_cycle(LF)
        RF_stance, RF_swing = count_stance_swing_per_cycle(RF)
        LM_stance, LM_swing = count_stance_swing_per_cycle(LM)
        RM_stance, RM_swing = count_stance_swing_per_cycle(RM)
        LH_stance, LH_swing = count_stance_swing_per_cycle(LH)
        RH_stance, RH_swing = count_stance_swing_per_cycle(RH)
        # Add the stance and swing durations to the lists
        front_stance.append(LF_stance)
        front_stance.append(RF_stance)
        middle_stance.append(LM_stance)
        middle_stance.append(RM_stance)
        hind_stance.append(LH_stance)
        hind_stance.append(RH_stance)
        front_swing.append(LF_swing)
        front_swing.append(RF_swing)
        middle_swing.append(LM_swing)
        middle_swing.append(RM_swing)
        hind_swing.append(LH_swing)
        hind_swing.append(RH_swing)
  
    # Concatenate the lists
    front_stance = np.concatenate(front_stance)
    middle_stance = np.concatenate(middle_stance)
    hind_stance = np.concatenate(hind_stance)
    front_swing = np.concatenate(front_swing)
    middle_swing = np.concatenate(middle_swing)
    hind_swing = np.concatenate(hind_swing)
    print(f'Front stance: {front_stance}, Middle stance: {len(middle_stance)}, Hind stance: {len(hind_stance)}')

    # Convert time step to seconds
    front_stance, middle_stance, hind_stance = front_stance / 200, middle_stance / 200, hind_stance / 200
    front_swing, middle_swing, hind_swing = front_swing / 200, middle_swing / 200, hind_swing / 200

    max_duration = max(max(front_stance), max(middle_stance), max(hind_stance),
                          max(front_swing), max(middle_swing), max(hind_swing))
    min_duration = min(min(front_stance), min(middle_stance), min(hind_stance),
                            min(front_swing), min(middle_swing), min(hind_swing))

    # boxplot
    plt.figure(figsize=(5, 3))
    plt.subplot(1, 2, 1)
    plt.boxplot([front_stance, middle_stance, hind_stance], labels=['Front', 'Middle', 'Hind'])
    plt.title('Stance Duration', fontsize = 12)
    plt.ylabel('Duration (s)', fontsize = 12)
    plt.ylim(min_duration*1.2, max_duration*1.2)
    plt.subplot(1, 2, 2)
    plt.boxplot([front_swing, middle_swing, hind_swing], labels=['Front', 'Middle', 'Hind'])
    plt.title('Swing Duration', fontsize = 12)
    plt.ylabel('Duration (s)', fontsize = 12)
    plt.ylim(min_duration*1.2, max_duration*1.2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(title, fontsize = 14)
    plt.savefig(f'expert_data_builder/stick_insect/morphology/{save_name}.png') 


if __name__ == "__main__":
    save_name_A = 'swing_stance_duration_Aretaon'
    title_A = 'Aretaon'
    path_A = [Aretaon_1_path, Aretaon_2_path, Areraon_3_path]
    plot_swing_stance_duration(path_A, title_A, save_name_A)

    save_name_C = 'swing_stance_duration_Carausius'
    title_C = 'Carausius'
    path_C = [Carausius_1_path, Carausius_2_path, Carausius_3_path]
    plot_swing_stance_duration(path_C, title_C, save_name_C)

    save_name_M = 'swing_stance_duration_Medauroidea'
    title_M = 'Medauroidea'
    path_M = [Medauroidea_1_path, Medauroidea_2_path, Medauroidea_3_path]
    plot_swing_stance_duration(path_M, title_M, save_name_M)

