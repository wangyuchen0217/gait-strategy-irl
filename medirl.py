import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from plot_train import *

class DeepIRLFC(nn.Module):
    def __init__(self, n_input, n_h1=400, n_h2=300):
        super(DeepIRLFC, self).__init__()
        self.fc1 = nn.Linear(n_input, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.reward = nn.Linear(n_h2, 1)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.reward(x)
        return x

    def get_rewards(self, states):
        with torch.no_grad():
            rewards = self.forward(states)
        return rewards


def deep_maxent_irl(feature_matrix, transition_probability, discount, 
                    trajectories, learning_rate, epochs, n_bins, labels, test_folder, device):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
    inputs:
        feature_matrix    NxD matrix - the features for each state
        transition_probability         NxNxN_ACTIONS matrix - P_a[s0, a, s1] is the transition prob of 
                                        landing at state s1 when taking action 
                                        a at state s0
        discount       float - RL discount factor
        trajectories       a list of demonstrations
        learning_rate          float - learning rate
        epochs     int - number of optimization steps
    returns
        rewards     Nx1 vector - recoverred state rewards
    """
    print(f"Device: {device}")
    print("Starting IRL:")
    start_time = time.time()

    n_states, _, _ = transition_probability.shape

    # Initialize neural network model
    nn_r = DeepIRLFC(feature_matrix.shape[1], 3, 3).to(device)
    optimizer = optim.SGD(nn_r.parameters(), lr=learning_rate)

    # Find state visitation frequencies using demonstrations
    svf = demo_svf(trajectories, n_states)
    svf = torch.tensor(svf, dtype=torch.float32).to(device)

    # Training
    mean_rewards = []
    losses = []
    for i in range(epochs):
        # Compute the reward matrix
        rewards = nn_r.get_rewards(feature_matrix).squeeze()

        # Compute policy
        _, policy = value_iteration(transition_probability, rewards, discount, device, error=0.01, deterministic=False)

        # Compute expected state visitation frequencies
        expected_svf = compute_state_visition_freq(transition_probability, trajectories, policy, device, deterministic=False)

        # Compute gradients on rewards
        grad_r = svf - expected_svf

        # Apply gradients to the neural network
        optimizer.zero_grad()
        loss = -torch.sum(rewards * grad_r)  # Negative sign because we want to maximize
        l2_loss = sum(param.pow(2.0).sum() for param in nn_r.parameters())
        loss += l2_loss * 10  # L2 regularization
        print(f"Epoch {i + 1}/{epochs} - Loss: {loss.cpu().item()}")
        losses.append(loss.cpu().item())
        loss.backward()
        optimizer.step()

        # record the mean reward
        mean_reward = torch.mean(rewards).item()
        mean_rewards.append(mean_reward)
        plt.figure(figsize=(10, 8))
        plt.plot(mean_rewards)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Reward')
        plt.title('Training Progress')
        plt.savefig(test_folder+'mean_rewards.png')
        plt.close()
        # record the loss
        plt.figure(figsize=(10, 8))
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.savefig(test_folder+'losses.png')
        plt.close()

        # Print progress every 10 epochs
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {i + 1}/{epochs} - Time elapsed: {elapsed_time:.2f}s")
            r_np = normalize(rewards.cpu().numpy())
            if len(n_bins) == 2:
                plot_training_rewards_2d(r_np, n_bins, labels, str(i + 1), test_folder)
            elif len(n_bins) == 4:
                plot_training_rewards_4d(r_np, n_bins, labels, str(i + 1), test_folder)
            torch.save(rewards, test_folder + 'inferred_rewards' + str(i + 1) + '.pt')

    rewards = torch.tensor(r_np, dtype=torch.float32, device=device)
    return rewards


def compute_state_visition_freq(transition_probability, trajectories, policy, device, deterministic=False):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming
    inputs:
        transition_probability     NxNxN_ACTIONS matrix - transition dynamics
        trajectories   list of list of Steps - collected from expert
        policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy
    returns:
        expected_svf       Nx1 vector - state visitation frequencies
    """
    n_states, _, _ = transition_probability.shape

    trajectory_length = trajectories.shape[1]
    expected_svf = torch.zeros([n_states, trajectory_length], dtype=torch.float32, device=device)

    for traj in trajectories:
        expected_svf[traj[0, 0], 0] += 1
    expected_svf[:, 0] = expected_svf[:, 0] / len(trajectories)

    for t in range(trajectory_length - 1):
        if deterministic:
            for s in range(n_states):
                expected_svf[s, t + 1] = torch.sum(expected_svf[:, t] * transition_probability[:, int(policy[s]), s])
        else:
            for s in range(n_states):
                expected_svf[s, t + 1] = torch.sum(torch.sum(expected_svf[:, t].unsqueeze(1) * transition_probability[:, :, s] * policy, dim=1))
    expected_svf = torch.sum(expected_svf, dim=1)
    return expected_svf


def demo_svf(trajectories, n_states):
    """
    compute state visitation frequences from demonstrations
    input:
        trajectories   list of list of Steps - collected from expert
    returns:
        svf       Nx1 vector - state visitation frequences   
    """
    svf = torch.zeros(n_states, dtype=torch.float32)
    for traj in trajectories:
        for step in traj:
            svf[step[0]] += 1
    svf = svf / len(trajectories)
    return svf


def value_iteration(transition_probability, rewards, discount, device, error=0.01, deterministic=False):
    """
    Static value iteration function.
    """
    n_states, n_actions, _ = transition_probability.shape
    values = torch.zeros(n_states, dtype=torch.float32, device=device)

    while True:
        values_tmp = values.clone()
        for s in range(n_states):
            values[s] = torch.max(torch.stack([torch.sum(transition_probability[s, a, :] * (rewards[s] + discount * values_tmp)) for a in range(n_actions)]))
        if torch.max(torch.abs(values - values_tmp)) < error:
            break

    if deterministic:
        policy = torch.zeros(n_states, dtype=torch.long, device=device)
        for s in range(n_states):
            policy[s] = torch.argmax(torch.stack([torch.sum(transition_probability[s, a, :] * (rewards[s] + discount * values)) for a in range(n_actions)]))
        return values, policy
    else:
        policy = torch.zeros([n_states, n_actions], dtype=torch.float32, device=device)
        for s in range(n_states):
            v_s = torch.tensor([torch.sum(transition_probability[s, a, :] * (rewards[s] + discount * values)) for a in range(n_actions)]).to(device)
            policy[s, :] = v_s / torch.sum(v_s)
        return values, policy


def normalize(x):
    min_val, max_val = np.min(x), np.max(x)
    if min_val == max_val:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)
