"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.
The GPU version of the value iteration algorithm.
"""

import numpy as np
import torch

def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, device, threshold=1e-4):
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
                max_v = max(max_v, torch.dot(tp, reward + discount * v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount, device,
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

    reward = reward.to(device)
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, device, threshold=1e-4)
    else:
        v = v.to(device)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = torch.zeros((n_states, n_actions), device=device)
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :].to(device)
                Q[i, j] = torch.dot(p, reward + discount * v)
        Q = Q - Q.max(dim=1, keepdim=True).values  # For numerical stability.
        Q = torch.exp(Q) / torch.exp(Q).sum(dim=1, keepdim=True)
        return Q
    
    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: torch.dot(
                       transition_probabilities[s, a, :].to(device),
                       reward + discount * v
                   ).item())
    policy = torch.tensor([_policy(s) for s in range(n_states)], device=device)
    return policy

