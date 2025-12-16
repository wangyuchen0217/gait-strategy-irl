'''
This code is used to plot the walking dynamic data (velocity, direction, acceleration)
including the histogram of the binned data, the curve of the continuous data.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_bins_histogram(data, title, xlabel, bin_step, savename='CarausuisC00_histogram_vel'):
    data = pd.DataFrame(data)
    count_per_bin = data.value_counts().sort_index()
    count_per_bin.index = range(len(count_per_bin.index))
    plt.figure(figsize=(4, 3))
    plt.bar(count_per_bin.index, count_per_bin.values, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('expert_demonstration/expert/plot/walking_dynamic_parameters/' + savename + '.png')


def plot_data_curve(vel, direction, acc, insect_state_name,
                    vel_binned, direction_binned, acc_binned):

    vel = vel[:300]
    direction = direction[:300]
    acc = acc[:300]
    vel_binned = vel_binned[:300]
    direction_binned = direction_binned[:300]
    acc_binned = acc_binned[:300]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 6))
    # -------- Velocity --------
    ax = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(vel, color='tab:blue', label='Data')
    l2, = ax2.plot(vel_binned, color='tab:orange', linestyle='-.', label='Binned Data')
    ax.set_title('Velocity', fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Value (mm/s)', fontsize=14)
    ax2.set_ylabel('Binned Value', fontsize=14)

    # -------- Direction --------
    ax = axes[1]
    ax2 = ax.twinx()
    l1, = ax.plot(direction, color='tab:blue', label='Data')
    l2, = ax2.plot(direction_binned, color='tab:orange', linestyle='-.', label='Binned Data')
    ax.set_title('Direction', fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Value (deg)', fontsize=14)
    ax2.set_ylabel('Binned Value', fontsize=14)

    # -------- Acceleration --------
    ax = axes[2]
    ax2 = ax.twinx()
    l1, = ax.plot(acc, color='tab:blue', label='Data')
    l2, = ax2.plot(acc_binned, color='tab:orange', linestyle='-.', label='Binned Data')
    ax.set_title('Acceleration', fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel(r'Value (mm/s$^2$)', fontsize=14)
    ax2.set_ylabel('Binned Value', fontsize=14)

    plt.tight_layout()
    plt.savefig(
        'expert_demonstration/expert/plot/walking_dynamic_parameters/'
        + insect_state_name + '_curve.png'
    )
