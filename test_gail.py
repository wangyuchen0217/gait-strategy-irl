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

# Load the trained policy
loaded_policy = PPO.load("trained_policy_gail")

# Create and wrap the environment
env = gym.make('StickInsect-v0')
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
# Reset the environment and get the initial observation
obs = env.reset()

# Initialize variables to store cumulative reward and done flag
cumulative_reward = 0
done = False

# Run the policy until the episode is done
while not done:
    action, _states = loaded_policy.predict(obs, deterministic=True)  # Get the action from the policy
    obs, rewards, done, info = env.step(action)  # Take the action in the environment
    cumulative_reward += rewards[0]  # Sum up the rewards

    # Optionally, render the environment to visualize what's happening
    env.render()  # Uncomment this if the environment supports rendering

print("Total reward for this episode:", cumulative_reward)

# Close the environment
env.close()