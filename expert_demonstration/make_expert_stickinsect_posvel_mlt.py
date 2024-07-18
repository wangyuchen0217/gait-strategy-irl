import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("./") # add the root directory to the python path
from envs import *
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import time
import yaml
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET


'''functions'''
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

def get_observation_state(subject:str):
    with open("configs/trail_details.json", "r") as f:
        trail_details = json.load(f)
    insect_name = trail_details[f"T{subject}"]["insect_name"]
    insect_number = trail_details[f"T{subject}"]["insect_number"]
    id_1 = trail_details[f"T{subject}"]["id_1"]
    id_2 = trail_details[f"T{subject}"]["id_2"]
    id_3 = trail_details[f"T{subject}"]["id_3"]
    joint_path = os.path.join("expert_data_builder/stick_insect", insect_name,
                                                    f"{insect_number}_{id_1}_{id_2}_{id_3}.csv")
    joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
    joint_movement = data_smooth(joint_movement) # smooth the data
    # FTi joint angle minus 90 degree
    joint_movement[:,-6:] = joint_movement[:,-6:] - 90

    dt = 0.005  # The timestep of your data
    # Calculate velocities and accelerations
    velocities = np.diff(joint_movement, axis=0) / dt
    # Pad the arrays to match the length of the original data
    velocities = np.vstack((velocities, np.zeros((1, velocities.shape[1])))) # [2459, 24]
    return joint_movement, velocities

'''main'''
# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

#  Set up simulation without rendering
model_name = config_data.get("model")
model_path = 'envs/assets/' + model_name + '.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

set_initial_state = False
if set_initial_state:
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
    data.qpos[-24:] = np.array(init_qpos_data.split()).astype(np.float64)

subjects = 12
obs_state = []
action = []
leg_geoms = ['LF_tibia_geom', 'LM_tibia_geom', 'LH_tibia_geom', 'RF_tibia_geom', 'RM_tibia_geom', 'RH_tibia_geom']
leg_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in leg_geoms]
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
contact_matrix = np.zeros((2459, len(leg_geoms)), dtype=int)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # set a camera <camera name="top" mode="fixed" pos="5 0 20" xyaxes="1 0 0 0 1 0"/>
    viewer.cam.lookat[0] = 5  # x-coordinate of the point to look at
    viewer.cam.lookat[1] = 0  # y-coordinate
    viewer.cam.lookat[2] = 0  # z-coordinate
    viewer.cam.distance = 20  # Camera distance from the lookat point
    viewer.cam.azimuth = 90  # Camera azimuth angle in degrees
    viewer.cam.elevation = -90  # Camera elevation angle in degrees
    for i in range(1, subjects + 1):
        subject_number = f"{i:02}"
        joint_movement, velocities = get_observation_state(subject_number)
        for j in range(len(joint_movement)):  # Run exactly 2459 frames
            if not viewer.is_running():  # Check if the viewer has been closed manually
                break
            # implement the joint angle data
            joint_angle = np.deg2rad(joint_movement[j])
            data.ctrl[:24] = joint_angle
            data.ctrl[24:] = velocities[j]
            mujoco.mj_step(model, data)
            viewer.sync()
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            # Manage timing to maintain a steady frame rate
            time.sleep(model.opt.timestep)

            state = np.hstack((data.qpos.copy()[:], # [-24:] joint angles, [:] w/ torso
                                                data.qvel.copy()[:])) # [-24:] joint velocities, [:] w/ torso
            # record the state of each step
            obs_state.append(state) # [2459,48] only joint angles and velocities, [2459, 61] w/ torso
            action.append(data.ctrl.copy()) # [2459, 48] only joint angles and velocities, [2459, 61] w/ torso
            
            # Record contact data
            # for i in range(data.ncon):
            #         contact = data.contact[i]
            #         geom1 = contact.geom1
            #         geom2 = contact.geom2
            #         # Check if the contact involves a leg geom and the floor
            #         for leg_index, leg_id in enumerate(leg_ids):
            #             if (geom1 == leg_id and geom2 == floor_id) or (geom1 == floor_id and geom2 == leg_id):
            #                 contact_matrix[j, leg_index] = 1  # Mark contact

            # record the initial position
            if j == 0:
                initail_pos = data.qpos.copy()
                initail_pos = initail_pos[:]
                print("initail_pos:", initail_pos.shape)
                print("initail_pos:", initail_pos)

# record observation state and action
obs_states = np.array(obs_state)
print("expert_demo:", obs_states.shape)
np.save("expert_demonstration/expert/StickInsect-32-obs.npy", obs_state)
actions = np.array(action)
print("actions:", actions.shape)
np.save("expert_demonstration/expert/StickInsect-32-act.npy", action)