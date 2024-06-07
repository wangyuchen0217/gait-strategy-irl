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

SEED = 42

# Create the environment
env = gym.make('StickInsect-v0')
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# Load the expert dataset
expert = np.load('expert/StickInsect-v0.npy', allow_pickle=True)

# Extract observations and "actions" (which are the next observations in this context)
observations = expert[0, :-1, :]  # Exclude the last step to avoid indexing error
actions = expert[0, 1:, :]        # Shift by one to get the "next" step as the action

# The last observation won't have a corresponding "next" action
next_observations = np.roll(observations, -1, axis=0)
next_observations[-1] = observations[-1]   # Handle boundary by replicating the last observation

dones = np.zeros(len(observations), dtype=bool)
dones[-1] = True  # Mark the last timestep as terminal

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
    batch_size=64,
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
)

# Create the GAIL trainer
gail_trainer = GAIL(
    demonstrations=transitions,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# evaluate the learner before training
env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# train the learner and evaluate again
gail_trainer.train(20000)  # Train for 800_000 steps to match expert.
env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# save the trained model
learner.save("trained_policy_gail")

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))