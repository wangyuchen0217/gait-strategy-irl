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

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# Load the trained policy
loaded_policy = PPO.load("trained_policy_gail")

# Create and wrap the environment
exclude_xy = config_data.get("exclude_xy")
env = gym.make('StickInsect-v0',  
               exclude_current_positions_from_observation=exclude_xy, 
               render_mode="human",
               max_episode_steps=3000)
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
# Reset the environment and get the initial observation
obs = env.reset()
print("Initial observation:", obs)

# Initialize variables to store cumulative reward and done flag
cumulative_reward = 0
done = False
step_count = 0

while not done:
    action, _states = loaded_policy.predict(obs, deterministic=True)  # Get the action from the policy
    obs, reward, done, info = env.step(action)  # Take the action in the environment
    cumulative_reward += reward  # Sum up the rewards
    print(f"Step: {step_count}, Reward: {reward}, Done: {done}")
    
    env.render()  
    step_count += 1

print("Total reward for this episode:", cumulative_reward)

# Close the environment
env.close()