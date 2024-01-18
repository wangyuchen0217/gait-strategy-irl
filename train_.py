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
from algorithms.maxent_irl_ import *

# feature parameters
state_dim = 12  # 12 joint positions
action_dim = 12  # 12 joint actuators
# expert demonstration parameters
traj_num = 33
traj_len = 1270
trajectories = np.load('expert_demo.npy') # (33, 1270, 2, 12)
# hyperparameters
learning_rate = 0.01
gamma = 0.9
epochs = 100
np.random.seed(1)

# train: maxent_irl
irl_agent = MaxEntIRL(trajectories, state_dim, action_dim, gamma, learning_rate, epochs)
learned_weights = irl_agent.maxent_irl()