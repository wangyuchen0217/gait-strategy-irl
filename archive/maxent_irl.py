from envs import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MaxEntIRL:
    def __init__(self, expert_data, state_dim, action_dim, gamma, learning_rate, epochs):
        self.expert_data = expert_data
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(state_dim)
        self.reward_history = []

    def compute_feature_expectations(self, trajectories):
        total_feature_expectations = np.zeros(self.state_dim)
        for trajectory in trajectories:
            for state in trajectory:
                total_feature_expectations += state
        return total_feature_expectations / len(trajectories)

##################################################
    def compute_state_action_visitation(self, trajectories):
        '''state_action_count = np.zeros([self.state_dim, trajectories[0].shape[0]])
        for trajectory in trajectories:
                state_action_count += trajectory'''
        state_action_count = np.zeros([self.state_dim, trajectories[:,0].shape[0]])
        i = 0
        for trajectory in trajectories:
                state_action_count[:,i] += trajectory
                i += 1
        return state_action_count / len(trajectories)
##################################################

    def compute_state_action_visitation(self, trajectories):
        # [s, t] is the prob of visiting state s at time t
        state_action_count = np.zeros([self.state_dim, trajectories[:,0].shape[0]])
        for trajectory in trajectories:
            for state_action_pair in trajectory:
                state_action_count += state_action_pair
        return state_action_count / len(trajectories)

    def compute_softmax_policy(self, weights, state_action):
        exponentiated_values = np.exp(weights.dot(state_action))
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
            # Concatenate state and action dimensions for policy computation
            state_action_visitation_combined = np.concatenate((state_action_visitation[:, :self.state_dim],
                                                               state_action_visitation[:, -self.action_dim:]), axis=1)
            # Compute policy using the current reward weights
            policy = np.array([self.compute_softmax_policy(self.weights, state_action) for state_action in state_action_visitation_combined])
            # Compute expected feature counts under the current policy
            expected_feature_counts = self.compute_expected_feature_counts(policy, state_action_visitation)
            # Compute the gradient of the reward function
            gradient = self.compute_gradient(expert_feature_expectations, expected_feature_counts)
            # Update the reward weights
            self.weights += self.learning_rate * gradient.mean(axis=0)
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
    
    def save_reward_history(self, save_path):
        reward_history = np.array(self.reward_history)
        pd.DataFrame(reward_history).to_csv(save_path, header=None, index=None)
    
    def save_learned_weights(self, save_path, format="npy"):
        if format == "npy":
            np.save(save_path, self.weights)
        elif format == "csv":
            learned_weights = np.array(self.weights)
            pd.DataFrame(learned_weights).to_csv(save_path, header=None, index=None)
