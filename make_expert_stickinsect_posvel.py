import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py    
import yaml
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET

# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

# smooth the data
def Kalman1D(observations,damping=1):
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.03
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
            )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def data_smooth(data):
    for i in range(data.shape[1]):
        smoothed_data = Kalman1D(data[:,i], damping=1).reshape(-1,1)
        data[:,i] = smoothed_data[:,0]
    return data


'''firl-stickinsect-v0'''
animal = "Carausius"
joint_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
joint_movement = data_smooth(joint_movement) # smooth the data

# FTi joint angle minus 90 degree
joint_movement[:,-6:] = joint_movement[:,-6:] - 90

dt = 0.005  # The timestep of your data
# Calculate velocities and accelerations
velocities = np.diff(joint_movement, axis=0) / dt
# Pad the arrays to match the length of the original data
velocities = np.vstack((velocities, np.zeros((1, velocities.shape[1])))) # [2459, 24]

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Parse the XML file to extract custom data
tree = ET.parse(model_path)
root = tree.getroot()
# Find the custom element and extract the init_qpos data
init_qpos_data = None
for custom in root.findall('custom'):
    for numeric in custom.findall('numeric'):
        if numeric.get('name') == 'init_qpos':
            init_qpos_data = numeric.get('data')
            break
sim.data.qpos[-24:] = np.array(init_qpos_data.split()).astype(np.float64)

trajecroty = []
torso_position = []
torques = []
for j in range(2459): # 2459 is the length of each trajectory

    # implement the joint angle data
    joint_angle = np.deg2rad(joint_movement[j])
    sim.data.ctrl[:24] = joint_angle
    sim.data.ctrl[24:] = velocities[j]
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos.copy()[-24:], 
                                        sim.get_state().qvel.copy()[-24:]))
    # record the state of each step
    trajecroty.append(state) # [2459,24]
    torso_position.append(sim.data.qpos[:3].copy()) # [2459,3]
    # get data of the torques sensor
    torques.append(sim.data.sensordata.copy())

    # record the initial position
    if j == 0:
        initail_pos = sim.get_state().qpos.copy()
        initail_pos = initail_pos[:]
        print("initail_pos:", initail_pos.shape)
        print("initail_pos:", initail_pos)

# record each trails
trajectories = np.array([trajecroty]) # [1, 2459, 24]
print("expert_demo:", trajectories.shape)
# np.save("StickInect-v0.npy", trajectories)

# Function to get the body ID associated with a sensor
def get_sensor_body_id(sensor_name):
    sensor_id = model.sensor_name2id(sensor_name)
    return model.sensor_objid[sensor_id]

# Function to transform torque from sensor frame to joint frame
def transform_torque(sensor_name, torque_vec):
    sensor_id = model.sensor_name2id(sensor_name)
    body_id = model.sensor_objid[sensor_id]
    # Get the rotation matrix from sensor frame to world frame
    sensor_rot = sim.data.get_site_xmat(sensor_name).reshape(3, 3)
    # Get the rotation matrix from world frame to joint frame
    body_rot = sim.data.get_body_xmat(body_id).reshape(3, 3)
    # Transform torque vector
    world_torque = np.dot(sensor_rot, torque_vec)
    joint_torque = np.dot(body_rot.T, world_torque)
    return joint_torque

# Function to process all timesteps of torque data
def process_all_timesteps(torques):
    all_transformed_torque = []
    for timestep_data in torques:
        transformed_torque_data = np.array([transform_torque(sensor_names[i], timestep_data[i]) for i in range(24)])
        all_transformed_torque.append(transformed_torque_data)
    return np.array(all_transformed_torque)

# record the torques
torques = np.array(torques) # [2459, 72]
torques = torques.reshape(-1, 24, 3) # [2459, 24, 3]
sensor_names =  ["LF_Sup_site","LM_Sup_site","LH_Sup_site","RF_Sup_site","RM_Sup_site","RH_Sup_site",
                                    "LF_CTr_site","LM_CTr_site","LH_CTr_site","RF_CTr_site","RM_CTr_site","RH_CTr_site",
                                    "LF_ThC_site","LM_ThC_site","LH_ThC_site","RF_ThC_site","RM_ThC_site","RH_ThC_site",
                                    "LF_FTi_site","LM_FTi_site","LH_FTi_site","RF_FTi_site","RM_FTi_site","RH_FTi_site"]
transformed_torques = process_all_timesteps(torques) 
print("transformed_torque_data:", transformed_torques.shape)

torque_save_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22_torques.csv")
pd.DataFrame(transformed_torques).to_csv(torque_save_path, index=False, header=None)

# record the torso position
# plt.figure()
# torso_position = np.array(torso_position)
# plt.plot(torso_position[:,0], torso_position[:,1])
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("c21_0680_trajectory_simulated")
# plt.grid()
# plt.show()
# # plt.savefig("c21_0680_002_3.png")