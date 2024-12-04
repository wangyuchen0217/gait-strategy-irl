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


def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters, device):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
    inputs:
        feat_map    NxD matrix - the features for each state
        P_a         NxNxN_ACTIONS matrix - P_a[s0, a, s1] is the transition prob of 
                                        landing at state s1 when taking action 
                                        a at state s0
        gamma       float - RL discount factor
        trajs       a list of demonstrations
        lr          float - learning rate
        n_iters     int - number of optimization steps
    returns
        rewards     Nx1 vector - recoverred state rewards
    """
    print(f"Device: {device}")
    print("Starting IRL:")
    start_time = time.time()

    N_STATES, N_ACTIONS, _ = P_a.shape

    # Initialize neural network model
    nn_r = DeepIRLFC(feat_map.shape[1], 3, 3).to(device)
    optimizer = optim.SGD(nn_r.parameters(), lr=lr)

    # Find state visitation frequencies using demonstrations
    mu_D = demo_svf(trajs, N_STATES)
    mu_D = torch.tensor(mu_D, dtype=torch.float32).to(device)

    # Training
    for i in range(n_iters):
        # Compute the reward matrix
        rewards = nn_r.get_rewards(feat_map).squeeze()

        # Compute policy
        _, policy = value_iteration(P_a, rewards, gamma, device, error=0.01, deterministic=False)

        # Compute expected state visitation frequencies
        mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, device, deterministic=False)

        # Compute gradients on rewards
        grad_r = mu_D - mu_exp

        # Apply gradients to the neural network
        optimizer.zero_grad()
        loss = -torch.sum(rewards * grad_r)  # Negative sign because we want to maximize
        l2_loss = sum(param.pow(2.0).sum() for param in nn_r.parameters())
        loss += l2_loss * 10  # L2 regularization
        loss.backward()
        optimizer.step()

        # Print progress every 10 epochs
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {i + 1}/{n_iters} - Time elapsed: {elapsed_time:.2f}s")

    rewards = nn_r.get_rewards(feat_map).cpu().numpy()
    return normalize(rewards)


def compute_state_visition_freq(P_a, gamma, trajs, policy, device, deterministic=False):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming
    inputs:
        P_a     NxNxN_ACTIONS matrix - transition dynamics
        gamma   float - discount factor
        trajs   list of list of Steps - collected from expert
        policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy
    returns:
        p       Nx1 vector - state visitation frequencies
    """
    N_STATES, N_ACTIONS, _ = P_a.shape

    T = trajs.shape[1]
    mu = torch.zeros([N_STATES, T], dtype=torch.float32, device=device)

    for traj in trajs:
        mu[traj[0, 0], 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)

    for t in range(T - 1):
        if deterministic:
            for s in range(N_STATES):
                mu[s, t + 1] = torch.sum(mu[:, t] * P_a[:, int(policy[s]), s])
        else:
            for s in range(N_STATES):
                mu[s, t + 1] = torch.sum(torch.sum(mu[:, t].unsqueeze(1) * P_a[:, :, s] * policy, dim=1))
    p = torch.sum(mu, dim=1)
    return p


def demo_svf(trajs, n_states):
    """
    compute state visitation frequences from demonstrations
    input:
        trajs   list of list of Steps - collected from expert
    returns:
        p       Nx1 vector - state visitation frequences   
    """
    p = torch.zeros(n_states, dtype=torch.float32)
    for traj in trajs:
        for step in traj:
            p[step[0]] += 1
    p = p / len(trajs)
    return p


def value_iteration(P_a, rewards, gamma, device, error=0.01, deterministic=False):
    """
    Static value iteration function.
    """
    N_STATES, N_ACTIONS, _ = P_a.shape
    values = torch.zeros(N_STATES, dtype=torch.float32, device=device)

    while True:
        values_tmp = values.clone()
        for s in range(N_STATES):
            values[s] = torch.max(torch.stack([torch.sum(P_a[s, a, :] * (rewards[s] + gamma * values_tmp)) for a in range(N_ACTIONS)]))
        if torch.max(torch.abs(values - values_tmp)) < error:
            break

    if deterministic:
        policy = torch.zeros(N_STATES, dtype=torch.long, device=device)
        for s in range(N_STATES):
            policy[s] = torch.argmax(torch.stack([torch.sum(P_a[s, a, :] * (rewards[s] + gamma * values)) for a in range(N_ACTIONS)]))
        return values, policy
    else:
        policy = torch.zeros([N_STATES, N_ACTIONS], dtype=torch.float32, device=device)
        for s in range(N_STATES):
            v_s = torch.tensor([torch.sum(P_a[s, a, :] * (rewards[s] + gamma * values)) for a in range(N_ACTIONS)]).to(device)
            policy[s, :] = v_s / torch.sum(v_s)
        return values, policy


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
