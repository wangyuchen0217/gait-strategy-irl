import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_bins_histogram(data, title, xlabel, bin_step, savename='CarausuisC00_histogram_vel'):
    data = pd.DataFrame(data)
    count_per_bin = data.value_counts().sort_index()
    count_per_bin.index = range(len(count_per_bin.index))
    plt.figure(figsize=(8, 6))
    plt.bar(count_per_bin.index, count_per_bin.values, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('expert_demonstration/expert/plot/' + savename + '.png')

def plot_data_curve(vel, direction, acc, insect_state_name):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    # Plot for velocity
    axes[0].plot(vel, color='#394E86', linewidth=2)
    axes[0].set_title(insect_state_name, fontsize=14)
    axes[0].set_xticks(axes[0].get_xticks())
    axes[0].set_yticks(axes[0].get_yticks())
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].set_xlabel('Time Steps', fontsize=14)
    axes[0].set_ylabel('Velocity', fontsize=14)
    # Plot for direction
    axes[1].plot(direction, color='#394E86', linewidth=2)
    axes[1].set_title(insect_state_name, fontsize=14)
    axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_yticks(axes[1].get_yticks())
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].set_xlabel('Time Steps', fontsize=14)
    axes[1].set_ylabel('Direction', fontsize=14)
    # Plot for acceleration
    axes[2].plot(acc, color='#394E86', linewidth=2)
    axes[2].set_title(insect_state_name, fontsize=14)
    axes[2].set_xticks(axes[2].get_xticks())
    axes[2].set_yticks(axes[2].get_yticks())
    axes[2].tick_params(axis='both', labelsize=14)
    axes[2].set_xlabel('Time Steps', fontsize=14)
    axes[2].set_ylabel('Acceleration', fontsize=14)
    plt.tight_layout()
    plt.savefig('expert_demonstration/expert/plot/'+insect_state_name+'_curve.png')