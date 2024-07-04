import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("./") # add the root directory to the python path
import envs

import torch
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.wrappers import RolloutInfoWrapper

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)
exclude_xy = config_data['exclude_xy']
SEED = config_data['env']['seed']
horizon = config_data['env']['horizon']

# Load the trained reward function
reward_net = torch.load("trained_policy_mce_irl.pth")

# Create and wrap the original environment
env = gym.make('StickInsect-v0',
               exclude_current_positions_from_observation=exclude_xy,
               max_episode_steps=horizon)
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# Setup PPO with the custom environment
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
    verbose=1
)

# Train the model
learner.learn(total_timesteps=10000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(learner, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# Save the trained model
learner.save("ppo_trained_with_mce_irl_reward.pth")

# Close the environment
env.close()
