import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


trajectories = torch.tensor(trajectories, device=device).float()
transition_probability = torch.tensor(transition_probability, device=device).float()
policy = torch.tensor(policy, device=device).float()

start_state_count = torch.zeros(n_states, device=device)
start_states = trajectories[:, 0, 0].long()
start_state_count.index_add_(0, start_states, torch.ones_like(start_states, dtype=torch.float, device=device))
p_start_state = start_state_count / n_trajectories

print(p_start_state)

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

print(expected_svf.sum(dim=1))