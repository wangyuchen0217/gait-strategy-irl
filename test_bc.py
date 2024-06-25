import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.data import types
import envs
import yaml
import xml.etree.ElementTree as ET
import mujoco
import torch

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# Load the trained policy
loaded_policy = torch.load("trained_policy_bc.pth")
loaded_policy = loaded_policy.policy # Accessing the policy attribute which is a PyTorch model
loaded_policy.eval() # Set the model to evaluation mode

# Create and wrap the environment
exclude_xy = config_data.get("exclude_xy")
env = gym.make('StickInsect-v0',  exclude_current_positions_from_observation=exclude_xy, render_mode="human")
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
# Reset the environment and get the initial observation
obs = env.reset()
# print("Initial observation:", obs)

# Initialize variables to store cumulative reward and done flag
cumulative_reward = 0
done = False
step_count = 0

while not done:
    # Convert the observation to tensor, and add batch dimension if necessary
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).squeeze(0)

    with torch.no_grad():  # Disable gradient calculation for inference
        action, _ = loaded_policy.predict(obs_tensor, deterministic=True)  # Get action and ignore additional outputs
    # print("Action:", action)
    # print("Action shape:", action.shape)
    # print("Action type:", type(action))
    # Convert the action from (48,) to (1, 48) to match the expected input shape
    action = action.reshape(1, -1)
    # print("Action shape after reshape:", action.shape)

    obs, reward, done, info = env.step(action)  # Take the action in the environment
    cumulative_reward += reward  # Sum up the rewards
    print(f"Step: {step_count}, Reward: {reward}, Done: {done}")
    
    env.render()  
    step_count += 1

print("Total reward for this episode:", cumulative_reward)

# Close the environment
env.close()