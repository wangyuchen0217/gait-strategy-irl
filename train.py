import gym
import numpy as np
from stable_baselines3 import PPO
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.util.logger import configure
from imitation.data.rollout import flatten_trajectories
import envs

# Create the environment
env = gym.make('StickInsect-v0')
env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# Load the expert dataset
trajectory_data = np.load('expert/StickInsect-v0.npy', allow_pickle=True)

# Extract observations and "actions" (which are the next observations in this context)
observations = trajectory_data[0, :-1, :]  # Exclude the last step to avoid indexing error
actions = trajectory_data[0, 1:, :]        # Shift by one to get the "next" step as the action

# The last observation won't have a corresponding "next" action
next_observations = np.roll(observations, -1, axis=0)
next_observations[-1] = observations[-1]   # Handle boundary by replicating the last observation

dones = np.zeros(len(observations), dtype=bool)
dones[-1] = True  # Mark the last timestep as terminal

# Prepare the trajectories dictionary
trajectories = {
    'obs': observations,
    'acts': actions,
    'next_obs': next_observations,
    'dones': dones,
    'infos': [{} for _ in range(len(observations))]
}

# Flatten the trajectories if needed
# transitions = flatten_trajectories([trajectories])

# Initialize the PPO model (generator)
gen_algo = PPO('MlpPolicy', env, verbose=1)

# Configure the logger
logger = configure(folder="output", format_strs=["stdout", "csv", "tensorboard"])

# Initialize the GAIL model
gail_trainer = GAIL(demonstrations=trajectories, venv=env, gen_algo=gen_algo, expert_batch_size=32, gen_batch_size=32, logger=logger)

# Train the model
gail_trainer.train(total_timesteps=10000)

# Save the model
gen_algo.save("gail_stickinsect")

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = gen_algo.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


