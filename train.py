import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from envs import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco_py
from sklearn.preprocessing import MinMaxScaler
import yaml
import datetime
import dateutil.tz
from algorithms.maxent_irl import *

# feature parameters
state_num = 1270*33  # 33 trajectories, 1270 timesteps each
action_num = 1270*33  # 33 trajectories, 1270 timesteps each
state_dim = 12  # 12 joint positions
action_dim = 12  # 12 joint actuators
feature_matrix = np.zeros((state_num, state_dim))
# expert demonstration parameters
traj_num = 33
traj_len = 1270
trajectories = np.load('expert_demo.npy') # (33, 1270, 2, 12)
observed_states = np.random.rand(traj_num, traj_len, state_num)
observed_actions = np.random.rand(traj_num, traj_len, action_dim)
# hyperparameters
learning_rate = 0.01
discount_factor = 0.9
transition_probability = np.random.rand(state_num, state_num, state_num) # (12, 12, 12)
epochs = 100
np.random.seed(1)

# train: maxent_irl
reward = maxent_irl(feature_matrix, action_num, discount_factor, transition_probability, 
                                        trajectories, epochs, learning_rate)