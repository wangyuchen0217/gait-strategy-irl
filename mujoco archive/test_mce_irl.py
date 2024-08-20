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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)
exclude_xy = config_data['exclude_xy']

# Load the expert dataset and fit PCA and scaler
obs_states = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-obs.npy', allow_pickle=True)
observations = obs_states[0, :-1, 2:] if exclude_xy else obs_states[0, :-1, :]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(observations)

pca_dimension = config_data['irl']['pca_dimension']
pca = PCA(n_components=pca_dimension)
pca.fit(scaled_data)

# Load the trained policy
loaded_policy = PPO.load("trained_policy_ppo.pth")

# Create and wrap the environment
env = gym.make('StickInsect-v0-disc',  
                pca=pca,
                scaler=scaler,
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