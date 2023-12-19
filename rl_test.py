import gym
from rl import QLearningAgent 
import numpy as np


learned_weights = np.load("learned_weights.npy")
env = gym.make('CricketEnv2D-v0')

# 4. Train RL Agent
rl_epochs = 1000
rl_agent = QLearningAgent(env, learned_weights)

for epoch in range(rl_epochs):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        reward = learned_weights.dot(state)
        action = rl_agent.select_action(state)
        next_state, _, done, _ = env.step(action)

        rl_agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    print(f"Epoch {epoch}, Total Reward: {total_reward}")

# Optionally, test the RL agent
test_episodes = 10
for _ in range(test_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = rl_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Test Episode, Total Reward: {total_reward}")

env.close()
