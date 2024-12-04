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
                    trajectories, learning_rate, epochs, device):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
    inputs:
        feature_matrix    NxD matrix - the features for each state
        transition_probability         NxNxN_ACTIONS matrix - P_a[s0, a, s1] is the transition prob of 
                                        landing at state s1 when taking action 
                                        a at state s0
        discount       float - RL discount factor
        trajectories       a list of demonstrations
        lr          float - learning rate
        n_iters     int - number of optimization steps
    returns
        rewards     Nx1 vector - recoverred state rewards
    """
    print(f"Device: {device}")
    print("Starting IRL:")
    start_time = time.time()

    N_STATES, N_ACTIONS, _ = transition_probability.shape

    # Initialize neural network model
    nn_r = DeepIRLFC(feature_matrix.shape[1], 3, 3).to(device)
    optimizer = optim.SGD(nn_r.parameters(), lr=learning_rate)

    # Find state visitation frequencies using demonstrations
    svf = demo_svf(trajectories, N_STATES)
    svf = torch.tensor(svf, dtype=torch.float32).to(device)

    # Training
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
        loss.backward()
        optimizer.step()

        # Print progress every 10 epochs
        if (i + 1) % 1 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {i + 1}/{epochs} - Time elapsed: {elapsed_time:.2f}s")

    rewards = nn_r.get_rewards(feature_matrix).cpu().numpy()
    return normalize(rewards)


def compute_state_visition_freq(transition_probability, trajectories, policy, device, deterministic=False):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming
    inputs:
        transition_probability     NxNxN_ACTIONS matrix - transition dynamics
        gamma   float - discount factor
        trajectories   list of list of Steps - collected from expert
        policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy
    returns:
        p       Nx1 vector - state visitation frequencies
    """
    N_STATES, N_ACTIONS, _ = transition_probability.shape

    trajectory_length = trajectories.shape[1]
    expected_svf = torch.zeros([N_STATES, trajectory_length], dtype=torch.float32, device=device)

    for traj in trajectories:
        expected_svf[traj[0, 0], 0] += 1
    expected_svf[:, 0] = expected_svf[:, 0] / len(trajectories)

    for t in range(trajectory_length - 1):
        if deterministic:
            for s in range(N_STATES):
                expected_svf[s, t + 1] = torch.sum(expected_svf[:, t] * transition_probability[:, int(policy[s]), s])
        else:
            for s in range(N_STATES):
                expected_svf[s, t + 1] = torch.sum(torch.sum(expected_svf[:, t].unsqueeze(1) * transition_probability[:, :, s] * policy, dim=1))
    p = torch.sum(expected_svf, dim=1)
    return p


def demo_svf(trajectories, n_states):
    """
    compute state visitation frequences from demonstrations
    input:
        trajectories   list of list of Steps - collected from expert
    returns:
        p       Nx1 vector - state visitation frequences   
    """
    p = torch.zeros(n_states, dtype=torch.float32)
    for traj in trajectories:
        for step in traj:
            p[step[0]] += 1
    p = p / len(trajectories)
    return p


def value_iteration(transition_probability, rewards, discount, device, error=0.01, deterministic=False):
    """
    Static value iteration function.
    """
    N_STATES, N_ACTIONS, _ = transition_probability.shape
    values = torch.zeros(N_STATES, dtype=torch.float32, device=device)

    while True:
        values_tmp = values.clone()
        for s in range(N_STATES):
            values[s] = torch.max(torch.stack([torch.sum(transition_probability[s, a, :] * (rewards[s] + discount * values_tmp)) for a in range(N_ACTIONS)]))
        if torch.max(torch.abs(values - values_tmp)) < error:
            break

    if deterministic:
        policy = torch.zeros(N_STATES, dtype=torch.long, device=device)
        for s in range(N_STATES):
            policy[s] = torch.argmax(torch.stack([torch.sum(transition_probability[s, a, :] * (rewards[s] + discount * values)) for a in range(N_ACTIONS)]))
        return values, policy
    else:
        policy = torch.zeros([N_STATES, N_ACTIONS], dtype=torch.float32, device=device)
        for s in range(N_STATES):
            v_s = torch.tensor([torch.sum(transition_probability[s, a, :] * (rewards[s] + discount * values)) for a in range(N_ACTIONS)]).to(device)
            policy[s, :] = v_s / torch.sum(v_s)
        return values, policy


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
