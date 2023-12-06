import gym
from envs import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import mujoco_py
from MaxEnt_IRL import MaxEntIRL
from sklearn.preprocessing import MinMaxScaler

def normalization(data):
    data_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
    data_scaled = data_scaler.transform(data)
    # de-normalization
    # predict = scaler.inverse_transform(all_data_scaled)
    return data_scaler, data_scaled

# Load joint angle data from the CSV file
csv_file_path = 'Expert_data_builder/demo_dataset.csv'  
dataset = pd.read_csv(csv_file_path, header=0, usecols=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()
# normalize the dataset by each column
for i in range(dataset.shape[1]):
    scaled_data = normalization(dataset[:,i].reshape(-1,1)
    print("scaled_data:", scaled_data.shape)

#  check the shape of the dataset
print("dataset shape:", dataset.shape)
print("len(dataset):", len(dataset))

# Define the MuJoCo model and set initial state
model = mujoco_py.load_model_from_path('envs/assets/Cricket2D.xml')  
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Apply joint angles from the CSV data to the MuJoCo model
for i in range(len(dataset)):
    joint_angle = np.deg2rad(dataset[i])
    sim.data.ctrl[:] = joint_angle
    sim.step()
    viewer.render()

viewer.close()

# Now, you can use the final state of the MuJoCo simulation, 
# or continue interacting with the environment as needed
#final_state = sim.get_state()


