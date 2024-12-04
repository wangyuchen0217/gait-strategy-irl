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


def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False):
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
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    T = len(trajs[0])
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)

    for s in range(N_STATES):
        for t in range(T - 1):
            if deterministic:
                mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu[s, t + 1] = sum([sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
    p = np.sum(mu, 1)
    return p


def demo_svf(trajs, n_states):
    """
    compute state visitation frequences from demonstrations
    input:
        trajs   list of list of Steps - collected from expert
    returns:
        p       Nx1 vector - state visitation frequences   
    """
    p = np.zeros(n_states)
    for traj in trajs:
        for step in traj:
            p[step.cur_state] += 1
    p = p / len(trajs)
    return p


def value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True):
    """
    Static value iteration function.
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    values = np.zeros([N_STATES])

    while True:
        values_tmp = values.copy()
        for s in range(N_STATES):
            values[s] = max([sum([P_a[s, s1, a] * (rewards[s] + gamma * values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
        if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
            break

    if deterministic:
        policy = np.zeros([N_STATES])
        for s in range(N_STATES):
            policy[s] = np.argmax([sum([P_a[s, s1, a] * (rewards[s] + gamma * values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
        return values, policy
    else:
        policy = np.zeros([N_STATES, N_ACTIONS])
        for s in range(N_STATES):
            v_s = np.array([sum([P_a[s, s1, a] * (rewards[s] + gamma * values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
            policy[s, :] = v_s / np.sum(v_s)
        return values, policy


def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters, device):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL) using PyTorch.
    """
    print(f"Device: {device}")
    print("Starting IRL:")
    start_time = time.time()

    N_STATES, _, N_ACTIONS = np.shape(P_a)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize neural network model
    nn_r = DeepIRLFC(feat_map.shape[1], 3, 3).to(device)
    optimizer = optim.SGD(nn_r.parameters(), lr=lr)
    feat_map = torch.tensor(feat_map, dtype=torch.float32).to(device)

    # Find state visitation frequencies using demonstrations
    mu_D = demo_svf(trajs, N_STATES)
    mu_D = torch.tensor(mu_D, dtype=torch.float32).to(device)

    # Training
    for i in range(n_iters):
        # Compute the reward matrix
        rewards = nn_r.get_rewards(feat_map).squeeze()

        # Compute policy
        rewards_np = rewards.cpu().numpy()
        _, policy = value_iteration(P_a, rewards_np, gamma, error=0.01, deterministic=True)

        # Compute expected state visitation frequencies
        mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
        mu_exp = torch.tensor(mu_exp, dtype=torch.float32).to(device)

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
        if (i + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {i + 1}/{n_iters} - Time elapsed: {elapsed_time:.2f}s")

    rewards = nn_r.get_rewards(feat_map).cpu().numpy()
    return normalize(rewards)


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
