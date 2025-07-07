import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Aretaon_data_path = 'expert_data_builder/stick_insect/Aretaon/Animal11_100723_00_01_morphology.csv'
Carausius_data_path = 'expert_data_builder/stick_insect/Carausius/Animal12_110415_00_22_morphology.csv'
Medauroidea_data_path = 'expert_data_builder/stick_insect/Medauroidea/Animal06_110919_00_15_morphology.csv'

def load_data(file_path):
    data = pd.read_csv(file_path, header=[0], index_col=None)
    # calculate the torso length
    thorax = data['T1'].values.reshape(-1, 1) + data['T2'].values.reshape(-1, 1) + data['T3'].values.reshape(-1, 1)
    # calculate the leg length and reshape to a number
    LF = data['LF_fem'].values.reshape(-1, 1) + data['LF_tib'].values.reshape(-1, 1)
    LM = data['LM_fem'].values.reshape(-1, 1) + data['LM_tib'].values.reshape(-1, 1)
    LH = data['LH_fem'].values.reshape(-1, 1) + data['LH_tib'].values.reshape(-1, 1)
    RF = data['RF_fem'].values.reshape(-1, 1) + data['RF_tib'].values.reshape(-1, 1)
    RM = data['RM_fem'].values.reshape(-1, 1) + data['RM_tib'].values.reshape(-1, 1)
    RH = data['RH_fem'].values.reshape(-1, 1) + data['RH_tib'].values.reshape(-1, 1)
    # calculate the limb-to-torso ratio
    limb2thorax_ratio = {
        'LF': LF / thorax,
        'LM': LM / thorax,
        'LH': LH / thorax,
        'RF': RF / thorax,
        'RM': RM / thorax,
        'RH': RH / thorax
    }
    limb2thorax_mean = {
        'Front': (limb2thorax_ratio['LF'] + limb2thorax_ratio['RF']) / 2,
        'Middle': (limb2thorax_ratio['LM'] + limb2thorax_ratio['RM']) / 2,
        'Hind': (limb2thorax_ratio['LH'] + limb2thorax_ratio['RH']) / 2
    }
    limb2thorax_std = {
        'Front': np.std([limb2thorax_ratio['LF'], limb2thorax_ratio['RF']], axis=0),
        'Middle': np.std([limb2thorax_ratio['LM'], limb2thorax_ratio['RM']], axis=0),
        'Hind': np.std([limb2thorax_ratio['LH'], limb2thorax_ratio['RH']], axis=0)
    }
    return limb2thorax_mean, limb2thorax_std

def plot_limb2thorax(mean_dict, std_dict, species_name):
    labels = ['Front', 'Middle', 'Hind']
    means = [float(mean_dict[label]) for label in labels]
    stds = [float(std_dict[label]) for label in labels]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='skyblue', width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_title(f'Limb-to-Thorax Ratio: {species_name}', fontsize=22)
    
    # ax.set_ylim(0, max(np.array(means) + np.array(stds)) * 1.2)
    ax.set_ylim(0, 2.0) # set a fixed limit for better comparison
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig(f'expert_data_builder/stick_insect/morphology/limb2thorax_{species_name}.png')


if __name__ == "__main__":
    species_paths = {
        'Aretaon': Aretaon_data_path,
        'Carausius': Carausius_data_path,
        'Medauroidea': Medauroidea_data_path
    }

    for species, path in species_paths.items():
        mean, std = load_data(path)
        print(f"{species} Mean:", mean)
        print(f"{species} Std:", std)
        plot_limb2thorax(mean, std, species)

