import matplotlib.pyplot as plt
import numpy as np


def plot_training_rewards_2d(rewards, n_bin1, n_bin2, lable_bin1, lable_bin2, epoch, test_folder):
    state_rewards = rewards.reshape((n_bin2, n_bin1))
    plt.figure(figsize=(10, 8))
    plt.imshow(state_rewards, cmap='viridis', aspect='auto')
    plt.title("Grid-Based Reward Heatmap", fontsize=16)
    plt.xlabel(lable_bin1, fontsize=14)
    plt.ylabel(lable_bin2, fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+"reward_heatmap_"+ epoch +".png")

def plot_training_rewards_4d(rewards, n_bin1, n_bin2, n_bin3, n_bin4, lable_bin1, lable_bin2, lable_bin3, lable_bin4, epoch, test_folder):
    state_rewards = rewards.reshape((n_bin4, n_bin3, n_bin2, n_bin1))
    plt.figure(figsize=(10, 8))
    plt.imshow(state_rewards[0, 0, :, :], cmap='viridis', aspect='auto')
    plt.title("Training Reward Heatmap", fontsize=16)
    plt.xlabel(lable_bin1, fontsize=14)
    plt.ylabel(lable_bin2, fontsize=14)
    plt.colorbar(label='Reward Value')
    plt.savefig(test_folder+"reward_heatmap_"+ epoch +".png")

