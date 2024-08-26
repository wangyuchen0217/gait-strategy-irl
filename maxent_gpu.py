"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product

import numpy as np
import numpy.random as rn
import value_iteration
import time
import matplotlib.pyplot as plt
from plot_train import *
import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. Torch tensor
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: Torch tensor mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. Torch tensor with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    print("Starting IRL:")
    start_time = time.time()

    n_states, d_states = feature_matrix.shape

    # Initialise weights on GPU.
    alpha = torch.rand(d_states, device=device)

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix, trajectories)

    # Gradient descent on alpha.
    for i in range(epochs):
        r = feature_matrix @ alpha
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)
        grad = feature_expectations - feature_matrix.T @ expected_svf

        alpha += learning_rate * grad

        # Print progress every 10 epochs
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {i + 1}/{epochs} - Time elapsed: {elapsed_time:.2f}s")
            output = feature_matrix @ alpha
            torch.save(output.cpu(), f'acc_inferred_rewards_{i+1}.pt')

    return (feature_matrix @ alpha).cpu().numpy()

def find_feature_expectations(feature_matrix, trajectories):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. Torch tensor
                                    array with shape (N, D) where N is the number of states and D is the
                                    dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
                            are ints. Torch tensor with shape (T, L, 2) where T is the number of
                            trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = torch.zeros(feature_matrix.shape[1], device=device)

    for trajectory in trajectories:
        for state, _ in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. Torch tensor with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: Torch tensor mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. Torch tensor with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    policy = find_policy(n_states, n_actions, transition_probability, r, discount)

    start_state_count = torch.zeros(n_states, device=device)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count / n_trajectories

    expected_svf = torch.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                   policy[i, j] *
                                   transition_probability[i, j, k])

    return expected_svf.sum(axis=1)

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = torch.zeros(n_states, device=device)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, torch.dot(tp, reward + discount*v))

            new_diff = torch.abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = torch.zeros((n_states, n_actions), device=device)
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = torch.dot(p, reward + discount*v)
        Q -= Q.max(dim=1)[0].view(n_states, 1)  # For numerical stability.
        Q = torch.exp(Q)/torch.exp(Q).sum(axis=1).view(n_states, 1)
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: torch.sum(transition_probabilities[s, a, :] *
                                           (reward + discount * v)))

    policy = torch.tensor([_policy(s) for s in range(n_states)], device=device)
    return policy