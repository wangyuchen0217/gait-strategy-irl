'''
This code provides the function to plot the training rewards during IRL training.
Including the 2D and 4D reward heatmaps.
'''

import matplotlib.pyplot as plt


def plot_training_rewards_2d(rewards, n_bins, labels, epoch, test_folder):
    n_bin1, n_bin2 = n_bins[0], n_bins[1]
    label_bin1, label_bin2 = labels[0], labels[1]
    state_rewards = rewards.reshape((n_bin2, n_bin1))
    plt.figure(figsize=(10, 8))
    plt.imshow(state_rewards, cmap='viridis', aspect='auto')
    plt.title("Grid-Based Reward Heatmap", fontsize=16)
    plt.xlabel(label_bin1, fontsize=14)
    plt.ylabel(label_bin2, fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+'reward_heatmap_'+ epoch +'.png')
    plt.close()

def plot_training_rewards_4d(rewards, n_bins, labels, epoch, test_folder):
    n_bin1, n_bin2, n_bin3, n_bin4 = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    label_bin1, label_bin2, label_bin3, label_bin4 = labels[0], labels[1], labels[2], labels[3]
    state_rewards = rewards.reshape((n_bin4, n_bin3, n_bin2, n_bin1))
    plt.figure(figsize=(10, 8))
    plt.imshow(state_rewards[0, 0, :, :], cmap='viridis', aspect='auto')
    plt.title("Training Reward Heatmap", fontsize=16)
    plt.xlabel(label_bin1, fontsize=14)
    plt.ylabel(label_bin2, fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+'reward_heatmap_'+ epoch +'.png')
    plt.close()

