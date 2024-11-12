"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
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

    transition_probabilities = torch.tensor(transition_probabilities, device=device, dtype=torch.float32)
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
    # transition_probabilities = torch.tensor(transition_probabilities, device=device, dtype=torch.float32)
    transition_probabilities = transition_probabilities.clone().detach().to(device)
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
        return Q.cpu().numpy()
    
    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: torch.dot(
                       transition_probabilities[s, a, :].to(device),
                       reward + discount * v
                   ).item())
    policy = torch.tensor([_policy(s) for s in range(n_states)], device=device)
    return policy.cpu().numpy()

if __name__ == '__main__':
    # Quick unit test using gridworld.
    import mdp.gridworld as gridworld
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v = value([gw.optimal_policy_deterministic(s) for s in range(gw.n_states)],
              gw.n_states,
              gw.transition_probability,
              [gw.reward(s) for s in range(gw.n_states)],
              gw.discount)
    assert np.isclose(v,
                      [5.7194282, 6.46706692, 6.42589811,
                       6.46706692, 7.47058224, 7.96505174,
                       6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(gw.n_states,
                          gw.n_actions,
                          gw.transition_probability,
                          [gw.reward(s) for s in range(gw.n_states)],
                          gw.discount)
    assert np.isclose(v, opt_v).all()
