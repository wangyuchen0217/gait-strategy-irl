import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_histogram(data, title, xlabel, ylabel='Frequency', bins=30, savename='CarausuisC00_histogram_vel'):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig('expert_demonstration/expert/plot/' + savename + '.png')

def plot_bins_histogram(data, title, xlabel, bin_step, savename='CarausuisC00_histogram_vel'):
    data = pd.DataFrame(data)
    count_per_bin = data.value_counts().sort_index()
    count_per_bin.index = range(len(count_per_bin.index))
    plt.figure(figsize=(10, 6))
    plt.bar(count_per_bin.index, count_per_bin.values, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('expert_demonstration/expert/plot/' + savename + '.png')

def heatmap_direction_vel_reward(analysis_df):
    # Create a pivot table to count occurrences
    pivot_table = analysis_df.pivot_table(
                                        index='Velocity Bin', 
                                        columns='Direction Bin', 
                                        values='Gait Category',
                                        aggfunc=lambda x: x.value_counts().index[0],
                                        fill_value=0)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f")  # Change 'fmt' if needed for proper formatting
    plt.title('Heat Map of Gait Patterns by Velocity and Direction')
    plt.xlabel('Direction Bin')
    plt.ylabel('Velocity Bin')
    plt.show()

def heatmap_direction_vel_action(vel_binned, direction_binned):
    # Combine velocity and direction into a single DataFrame
    state_df = pd.DataFrame({
        'Velocity Bin': vel_binned.flatten(),
        'Direction Bin': direction_binned.flatten(),
    })

    # Create a pivot table to count occurrences of each state
    state_counts = state_df.pivot_table(index='Velocity Bin', columns='Direction Bin', aggfunc='size', fill_value=0)

    # Plot the heatmap of most accessed states
    plt.figure(figsize=(12, 8))
    sns.heatmap(state_counts, cmap='YlGnBu', annot=True, fmt="d")  # 'fmt="d"' for integer counts
    plt.title('Heat Map of Most Accessed States by Velocity and Direction')
    plt.xlabel('Direction Bin')
    plt.ylabel('Velocity Bin')
    plt.show()

def plot_states(vel_01, vel_02, vel_03, direction_01, direction_02, direction_03, acc_01, acc_02, acc_03, insect_state_name):
    # velocity
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    # Plot for vel_01
    axes[0].plot(vel_01)
    axes[0].set_title(insect_state_name, fontsize=14)
    axes[0].set_xticks(axes[0].get_xticks())
    axes[0].set_yticks(axes[0].get_yticks())
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].set_xlabel('Time Steps', fontsize=14)
    axes[0].set_ylabel('Vel', fontsize=14)
    axes[0].grid()
    # Plot for vel_02
    axes[1].plot(vel_02)
    axes[1].set_title(insect_state_name, fontsize=14)
    axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_yticks(axes[1].get_yticks())
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].set_xlabel('Time Steps', fontsize=14)
    axes[1].set_ylabel('Vel', fontsize=14)
    axes[1].grid()
    # Plot for vel_03
    axes[2].plot(vel_03)
    axes[2].set_title(insect_state_name, fontsize=14)
    axes[2].set_xticks(axes[2].get_xticks())
    axes[2].set_yticks(axes[2].get_yticks())
    axes[2].tick_params(axis='both', labelsize=14)
    axes[2].set_xlabel('Time Steps', fontsize=14)
    axes[2].set_ylabel('Vel', fontsize=14)
    axes[2].grid()
    plt.tight_layout()
    plt.savefig('expert_demonstration/expert/plot/'+insect_state_name+'_vel.png')

    # direction
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    # Plot for direction_01
    axes[0].plot(direction_01)
    axes[0].set_title(insect_state_name, fontsize=14)
    axes[0].set_xticks(axes[0].get_xticks())
    axes[0].set_yticks(axes[0].get_yticks())
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].set_xlabel('Time Steps', fontsize=14)
    axes[0].set_ylabel('Direction', fontsize=14)
    axes[0].grid()
    # Plot for direction_02
    axes[1].plot(direction_02)
    axes[1].set_title(insect_state_name, fontsize=14)
    axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_yticks(axes[1].get_yticks())
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].set_xlabel('Time Steps', fontsize=14)
    axes[1].set_ylabel('Direction', fontsize=14)
    axes[1].grid()
    # Plot for direction_03
    axes[2].plot(direction_03)
    axes[2].set_title(insect_state_name, fontsize=14)
    axes[2].set_xticks(axes[2].get_xticks())
    axes[2].set_yticks(axes[2].get_yticks())
    axes[2].tick_params(axis='both', labelsize=14)
    axes[2].set_xlabel('Time Steps', fontsize=14)
    axes[2].set_ylabel('Direction', fontsize=14)
    axes[2].grid()
    plt.tight_layout()
    plt.savefig('expert_demonstration/expert/plot/'+insect_state_name+'_direction.png')
    # acceleration
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    # Plot for acc_01
    axes[0].plot(acc_01)
    axes[0].set_title(insect_state_name, fontsize=14)
    axes[0].set_xticks(axes[0].get_xticks())
    axes[0].set_yticks(axes[0].get_yticks())
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].set_xlabel('Time Steps', fontsize=14)
    axes[0].set_ylabel('Acceleration', fontsize=14)
    axes[0].grid()
    # Plot for acc_02
    axes[1].plot(acc_02)
    axes[1].set_title(insect_state_name, fontsize=14)
    axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_yticks(axes[1].get_yticks())
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].set_xlabel('Time Steps', fontsize=14)
    axes[1].set_ylabel('Acceleration', fontsize=14)
    axes[1].grid()
    # Plot for acc_03
    axes[2].plot(acc_03)
    axes[2].set_title(insect_state_name, fontsize=14)
    axes[2].set_xticks(axes[2].get_xticks())
    axes[2].set_yticks(axes[2].get_yticks())
    axes[2].tick_params(axis='both', labelsize=14)
    axes[2].set_xlabel('Time Steps', fontsize=14)
    axes[2].set_ylabel('Acceleration', fontsize=14)
    axes[2].grid()
    plt.tight_layout()
    plt.savefig('expert_demonstration/expert/plot/'+insect_state_name+'_acc.png')

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