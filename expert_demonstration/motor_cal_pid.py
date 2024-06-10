import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
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

# calculate the torques
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    
    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def calculate_torques(desired_angles, current_angles, dt, pid_controllers):
    torques = []
    for i, pid in enumerate(pid_controllers):
        torque = pid.compute(desired_angles[i], current_angles[i], dt)
        torques.append(torque)
    return np.array(torques)


'''firl-stickinsect-v0'''
animal = "Carausius"
joint_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
joint_movement = data_smooth(joint_movement) # smooth the data

# FTi joint angle minus 90 degree
joint_movement[:,-6:] = joint_movement[:,-6:] - 90

# Initialize PID controllers for each joint
num_joints = joint_movement.shape[1]
pid_controllers = [PIDController(kp=30000, ki=0, kd=0) for _ in range(24)]

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
for j in range(2459): # 2459 is the length of each trajectory

    joint_angle = np.deg2rad(joint_movement[j])
    desired_angles = joint_angle
    current_angles = sim.data.qpos[-24:]  # Get current joint angles
    if j == 0:
        print("current_angles:", current_angles)
        print("desired_angles:", desired_angles)
    dt = sim.model.opt.timestep
    torques = calculate_torques(desired_angles, current_angles, dt, pid_controllers)
    
    # Apply torques to actuators
    sim.data.ctrl[:] = torques 
    sim.step()
    viewer.render()
    state = np.hstack((sim.get_state().qpos.copy()[-24:], 
                                        sim.get_state().qvel.copy()[-24:]))
    # record the state of each step
    trajecroty.append(state) # [2459,24]
    torso_position.append(sim.data.qpos[:3].copy()) # [2459,3]

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