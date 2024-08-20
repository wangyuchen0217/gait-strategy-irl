import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data
file_path = 'expert_demonstration/expert/CarausiusC00.csv'
data = pd.read_csv(file_path)

# Extract states (velocity and direction) and actions (gait)
states = data[['Velocity Bin', 'Direction Bin']].values
actions = data['Gait Category'].values

# Normalize the states if needed
scaler = MinMaxScaler()
states_normalized = scaler.fit_transform(states)

def reward_function(theta, state):
    # Define a simple linear reward function
    return np.dot(theta, state)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def feature_expectations(states, actions):
    # Compute feature expectations from the demonstration
    fe = np.zeros(states.shape[1])
    for i in range(len(states)):
        fe += states[i]
    return fe / len(states)

def maxent_irl(states, actions, feature_expectations, iterations=100, lr=0.01):
    theta = np.random.uniform(size=states.shape[1])
    
    for i in range(iterations):
        expected_state_visitation = np.zeros(states.shape[1])
        
        for s in range(len(states)):
            prob_action = softmax(reward_function(theta, states[s]))
            expected_state_visitation += prob_action * states[s]
        
        gradient = feature_expectations - expected_state_visitation / len(states)
        theta += lr * gradient

    return theta

def evaluate_reward(theta, state):
    return reward_function(theta, state)


# Calculate feature expectations from the demonstrations
fe = feature_expectations(states_normalized, actions)

# Run MaxEnt-IRL to learn the reward function
theta = maxent_irl(states_normalized, actions, fe)

# Evaluate the reward for a new state
new_state = np.array([3, 10])  # Example new state
reward = evaluate_reward(theta, new_state)
print(f"Reward for the new state: {reward}")


# Generate a grid of velocity and direction values
velocity_values = np.linspace(0, 1, 100)
direction_values = np.linspace(0, 1, 100)

# Initialize an empty grid for rewards
reward_grid = np.zeros((len(velocity_values), len(direction_values)))

# Evaluate the reward function for each combination of velocity and direction
for i, v in enumerate(velocity_values):
    for j, d in enumerate(direction_values):
        state = np.array([v, d])
        reward_grid[i, j] = reward_function(theta, state)

# Plot the heat map
plt.figure(figsize=(10, 8))
plt.imshow(reward_grid, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Reward')
plt.xlabel('Velocity')
plt.ylabel('Direction')
plt.title('Heat Map of Learned Reward Function')
plt.show()