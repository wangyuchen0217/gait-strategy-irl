import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.data import types
import envs

import yaml
import matplotlib.pyplot as plt
import logging
import torch

SEED = 42

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# Create the environment
exclude_xy = config_data.get("exclude_xy")
env = gym.make('StickInsect-v0',
               exclude_current_positions_from_observation=exclude_xy,
               max_episode_steps=3000)
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# Load the expert dataset
obs_states = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-obs.npy', allow_pickle=True)
actions = np.load('expert_demonstration/expert/StickInsect-v0-m3t-12-act.npy', allow_pickle=True)

# Extract observations and "actions" (which are the next observations in this context)
observations = obs_states[0, :-1, 2:] if exclude_xy else obs_states[0, :-1, :] # Exclude the last step to avoid indexing error
actions = actions[0, :-1, :] 

next_observations = obs_states[0, 1:, 2:] if exclude_xy else obs_states[0, 1:, :] # Exclude the first step to avoid indexing error

dones = np.zeros(len(observations), dtype=bool)
# dones[-1] = True  # Mark the last timestep as terminal

# transit the data to types.Transitions
transitions = types.Transitions(
    obs=observations,
    acts=actions,
    next_obs=next_observations,
    dones=dones,
    infos=[{} for _ in range(len(observations))]
)

# Create the learner (Proximal Policy Optimization)
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=16,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)

# Create the reward network
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

# Create the AIRL trainer
airl_trainer = AIRL(
    demonstrations=transitions,
    demo_batch_size=16,
    gen_replay_buffer_capacity=8,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True
)

# evaluate the learner before training
env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# train the learner and evaluate again
airl_trainer.train(800000)  # Train for 800_000 steps to match expert.
env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# save the trained model
# learner.save("trained_policy_airl")

# Extract the learned policy
learned_policy = airl_trainer.gen_algo.policy
# Save the trained policy
airl_trainer.gen_algo.save("trained_policy_airl")

# Extract the reward function from the discriminator
def reward_function(state, action):
    discriminator = airl_trainer.reward_trainer.discriminator
    logits = discriminator(torch.cat([state, action], dim=-1))
    reward = -logits  # or some transformation based on the discriminator output
    return reward


print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))