import gym
from envs import *
import numpy as np

# Import your MaxEnt IRL code or module
from MaxEnt_IRL import MaxEntIRL

# Step 2: Define and set up the Mujoco environment
env = gym.make('CricketEnv2D-v0')  
state_dim = env.observation_space.shape[0]

# Step 3: Collect expert demonstration data
def collect_expert_data(env, num_episodes=100):
    expert_data = []
    for _ in range(num_episodes):
        state = env.reset()
        trajectory = []
        done = False
        while not done:
            # Assuming the state directly provides joint angles
            joint_angles = state
            # Replace this with your own logic for selecting actions based on the current policy
            action = env.action_space.sample()
            # Take a step in the environment
            next_state, _, done, _ = env.step(action)
            # Collect joint angles for the expert data
            trajectory.append(joint_angles)
            state = next_state
        expert_data.append(trajectory)
    return expert_data

expert_data = collect_expert_data(env)

# Step 4: Use MaxEnt IRL with Mujoco Data
maxent_irl = MaxEntIRL(expert_data, state_dim)
learned_weights = maxent_irl.maxent_irl()

# Save the learned reward weights
maxent_irl.save_reward_weights("learned_reward_weights.npy")

# Plot training progress
maxent_irl.plot_training_progress()
