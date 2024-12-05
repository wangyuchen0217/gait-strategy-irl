"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product

import numpy as np
import numpy.random as rn
import torch
import value_iteration_gpu
import time
import matplotlib.pyplot as plt
from plot_train import *


def maxentirl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate, n_bins, labels, test_folder, device):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    # Initialize the start time
    print(f"Device: {device}")
    print("Starting IRL:")
    start_time = time.time()

    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    alpha = torch.rand(d_states, device=device)

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix,
                                                     trajectories, device)

    # Gradient descent on alpha.
    mean_rewards = []
    for i in range(epochs):
        # print("i: {}".format(i))
        r = torch.matmul(feature_matrix, alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories, device)
        grad = feature_expectations - torch.matmul(feature_matrix.T, expected_svf)

        alpha += learning_rate * grad
        rewards = torch.matmul(feature_matrix, alpha).reshape((n_states,))

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

        # Print progress every 10 epochs
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch {i + 1}/{epochs} - Time elapsed: {elapsed_time:.2f}s")
            if len(n_bins) == 2:
                plot_training_rewards_2d(rewards.cpu().numpy(), n_bins, labels, str(i + 1), test_folder)
            elif len(n_bins) == 4:
                plot_training_rewards_4d(rewards.cpu().numpy(), n_bins, labels, str(i + 1), test_folder)
            torch.save(rewards, test_folder + 'inferred_rewards' + str(i + 1) + '.pt')

    return rewards

def find_svf(n_states, trajectories, device):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
                            are ints. NumPy array with shape (T, L, 2) where T is the number of
                            trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = torch.zeros(n_states, device=device)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories, device):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
                                    array with shape (N, D) where N is the number of states and D is the
                                    dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
                            are ints. NumPy array with shape (T, L, 2) where T is the number of
                            trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = torch.zeros(feature_matrix.shape[1], device=device)

    # Using indexing to efficiently accumulate feature expectations.
    for trajectory in trajectories:
        states = trajectory[:, 0].long()
        feature_expectations += feature_matrix[states].sum(dim=0)

    feature_expectations /= len(trajectories)

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories, device):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy = value_iteration_gpu.find_policy(n_states, n_actions,
                                         transition_probability, r, discount, device)

    start_state_count = torch.zeros(n_states, device=device)
    start_states = trajectories[:, 0, 0].long()
    start_state_count.index_add_(0, start_states, torch.ones_like(start_states, dtype=torch.float, device=device))
    p_start_state = start_state_count / n_trajectories

    # Expected state visitation frequencies.
    expected_svf = torch.zeros((n_states, trajectory_length), device=device)
    expected_svf[:, 0] = p_start_state
    transition_probability = transition_probability.view(n_states, n_actions, n_states)
    # Iterate through each timestep
    for t in range(1, trajectory_length):
        # Propagate state visitation frequency to the next timestep
        # Compute the contribution of each state-action pair to all next states
        expected_svf_t_minus_1 = expected_svf[:, t - 1].unsqueeze(1).expand(-1, n_actions)  # Shape: (n_states, n_actions)
        # Multiply policy by previous visitation frequencies
        # Now we have the expected frequency of taking each action in each state
        expected_action_visitation = policy * expected_svf_t_minus_1  # Shape: (n_states, n_actions)
        # Sum over actions, effectively performing a batch matrix multiplication
        # For each state-action pair, determine how it contributes to next states
        expected_svf[:, t] = torch.einsum('sai,sa->i', transition_probability, expected_action_visitation)

    return expected_svf.sum(dim=1)

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = torch.max(torch.tensor([x1, x2]))
    min_x = torch.min(torch.tensor([x1, x2]))
    return max_x + torch.log(1 + torch.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(transition_probability[i, j, k] * V[k]
                           for k in range(n_states)))

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([transition_probability[i, j, k]
                          for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def expected_value_difference(n_states, n_actions, transition_probability,
    reward, discount, p_start_state, optimal_value, true_reward):
    """
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    reward: Reward vector mapping state int to reward. Shape (N,).
    discount: Discount factor. float.
    p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    """

    policy = value_iteration_gpu.find_policy(n_states, n_actions,
        transition_probability, reward, discount)
    value = value_iteration_gpu.value(policy.argmax(axis=1), n_states,
        transition_probability, true_reward, discount)

    evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
    return evd
