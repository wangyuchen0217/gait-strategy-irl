import numpy as np

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize Q-values
        self.Q = np.zeros((state_dim, action_dim))

    def choose_action(self, state):
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.Q[state, :])

    def update_q_values(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.Q[next_state, :])
        target = reward + self.discount_factor * self.Q[next_state, best_next_action]
        self.Q[state, action] += self.learning_rate * (target - self.Q[state, action])
