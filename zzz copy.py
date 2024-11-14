import numpy as np
from itertools import product


n_states = 4
n_actions = 3
trajectory_length = 3
n_trajectories = 3

trajectories = [
  [[0, 1], [1, 2], [2, 1]],  # From state 0 -> 1 -> 2
  [[2, 0], [3, 1], [1, 0]],  # From state 2 -> 3 -> 1
  [[1, 2], [0, 1], [3, 2]]   # From state 1 -> 0 -> 3
]

transition_probability = [
  [[0.4, 0.3, 0.3, 0.0], [0.2, 0.5, 0.3, 0.0], [0.1, 0.6, 0.2, 0.1]], # From State 0
  [[0.3, 0.3, 0.2, 0.2], [0.4, 0.1, 0.3, 0.2], [0.5, 0.0, 0.3, 0.2]], # From State 1
  [[0.1, 0.2, 0.5, 0.2], [0.3, 0.3, 0.3, 0.1], [0.2, 0.5, 0.2, 0.1]], # From State 2
  [[0.0, 0.4, 0.3, 0.3], [0.1, 0.4, 0.4, 0.1], [0.3, 0.2, 0.3, 0.2]]  # From State 3
]

policy = [
  [0.3, 0.4, 0.3],  # State 0: Probabilities for actions 0, 1, 2
  [0.2, 0.5, 0.3],  # State 1
  [0.4, 0.3, 0.3],  # State 2
  [0.5, 0.2, 0.3]   # State 3
]

trajectories = np.array(trajectories)
transition_probability = np.array(transition_probability)
policy = np.array(policy)

start_state_count = np.zeros(n_states)
for trajectory in trajectories:
    start_state_count[trajectory[0, 0]] += 1
p_start_state = start_state_count/n_trajectories

print(p_start_state)

expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
for t in range(1, trajectory_length):
    expected_svf[:, t] = 0
    for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
        expected_svf[k, t] += (expected_svf[i, t-1] *
                                policy[i, j] * # Stochastic policy
                                transition_probability[i, j, k])

print(expected_svf.sum(axis=1))