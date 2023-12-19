from envs import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MaxEntIRL:
    def __init__(self, expert_data, state_dim, epochs, learning_rate=0.01):
        self.expert_data = expert_data
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reward_history = []
        # Initialize the reward weights randomly
        self.weights = np.random.rand(state_dim)

    def compute_feature_expectations(self, trajectories):
        total_feature_expectations = np.zeros(self.state_dim)
        for trajectory in trajectories:
            for state in trajectory:
                total_feature_expectations += state
        return total_feature_expectations / len(trajectories)

    def compute_state_action_visitation(self, trajectories):
        state_action_count = np.zeros(self.state_dim)
        for trajectory in trajectories:
            for state in trajectory:
                state_action_count += state
        return state_action_count / len(trajectories)

    def compute_softmax_policy(self, weights, state):
        logits = weights.dot(state)
        max_logit = np.max(logits)
        exponentiated_values = np.exp(logits - max_logit)
        return exponentiated_values / np.sum(exponentiated_values)

    def compute_expected_feature_counts(self, policy, state_action_visitation):
        return policy * state_action_visitation[:, None]

    def compute_gradient(self, feature_expectations, expected_feature_counts):
        return feature_expectations - expected_feature_counts

    def maxent_irl(self):
        for epoch in range(self.epochs):
            # Compute state-action visitation frequency using the current policy
            state_action_visitation = self.compute_state_action_visitation(self.expert_data)
            # Compute feature expectations from the expert data
            expert_feature_expectations = self.compute_feature_expectations(self.expert_data)
            # Compute policy using the current reward weights
            policy = np.array([self.compute_softmax_policy(self.weights, state) for state in state_action_visitation])
            # Compute expected feature counts under the current policy
            expected_feature_counts = self.compute_expected_feature_counts(policy, state_action_visitation)
            # Compute the gradient of the reward function
            gradient = self.compute_gradient(expert_feature_expectations, expected_feature_counts)
            # Update the reward weights
            self.weights += self.learning_rate * gradient.mean(axis=0)
            # Save the reward weights
            self.reward_history.append(np.copy(self.weights))
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Reward Weights: {self.weights}")
        return self.weights
    
    def plot_training_progress(self, save_path):
        plt.figure()
        plt.plot(range(self.epochs), [np.linalg.norm(weights) for weights in self.reward_history])
        plt.xlabel("Epoch")
        plt.ylabel("Norm of Reward Weights")
        plt.title("Training Progress")
        plt.savefig(save_path)   
        
