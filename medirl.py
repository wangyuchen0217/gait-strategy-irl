import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(RewardNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.model(state)

def medirl_value(policy, n_states, transition_probabilities, reward_network, discount, device, threshold=1e-2):
    """
    Find the value function associated with a policy using a learned reward network.
    
    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to transition probabilities.
    reward_network: Neural network that approximates the reward function.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = torch.zeros(n_states, device=device)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            reward = reward_network(torch.tensor([s], dtype=torch.float32, device=device)).item()
            v[s] = sum(transition_probabilities[s, a, k] * (reward + discount * v[k]) for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def train_medirl(n_states, n_actions, transition_probabilities, trajectories, discount, device, epochs=100, learning_rate=0.01):
    """
    Train a MaxEnt Deep IRL model.
    
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to transition probabilities.
    trajectories: List of state-action pairs representing expert trajectories.
    discount: MDP discount factor. float.
    device: Device to run the training on (e.g., 'cuda' or 'cpu').
    epochs: Number of training epochs. int.
    learning_rate: Learning rate for the optimizer. float.
    """
    state_dim = n_states  # Assuming states are represented by their indices for simplicity.
    reward_network = RewardNetwork(state_dim).to(device)
    optimizer = optim.Adam(reward_network.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for trajectory in trajectories:
            for (state, action) in trajectory:
                state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
                predicted_reward = reward_network(state_tensor)

                # Calculate the expected value for the current state-action pair
                v = medirl_value([action for _ in range(n_states)], n_states, transition_probabilities, reward_network, discount, device)
                expected_value = v[state]

                # MaxEnt IRL loss is negative log-likelihood of the expert trajectories
                loss = -predicted_reward + discount * expected_value
                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")

    return reward_network

def find_policy_medirl(n_states, n_actions, transition_probabilities, reward_network, discount, device, threshold=1e-2, stochastic=True):
    """
    Find the optimal policy using a learned reward network.
    
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to transition probabilities.
    reward_network: Neural network that approximates the reward function.
    discount: MDP discount factor. float.
    device: Device to run the training on (e.g., 'cuda' or 'cpu').
    threshold: Convergence threshold, default 1e-2. float.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state (depending on stochasticity).
    """
    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = torch.zeros((n_states, n_actions), device=device)
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :].to(device)
                reward = reward_network(torch.tensor([i], dtype=torch.float32, device=device)).item()
                Q[i, j] = torch.dot(p, reward + discount * v)
        Q = Q - Q.max(dim=1, keepdim=True).values  # For numerical stability.
        Q = torch.exp(Q) / torch.exp(Q).sum(dim=1, keepdim=True)
        return Q

    policy = torch.tensor([torch.argmax(reward_network(torch.tensor([s], dtype=torch.float32, device=device))).item() for s in range(n_states)], device=device)
    return policy
