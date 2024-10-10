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


'''main'''
# open config file
with open("configs/irl.yml", "r") as f:
    config_data = yaml.safe_load(f)

animal = "Carausius"
joint_path = os.path.join("expert_data_builder/stick_insect", animal, 
                                                "Animal12_110415_00_22.csv")
joint_movement = pd.read_csv(joint_path, header=[0], index_col=None).to_numpy()
joint_movement = data_smooth(joint_movement) # smooth the data

# FTi joint angle minus 90 degree
joint_movement[:,-6:] = joint_movement[:,-6:] - 90
# simplized data
joint_movement = joint_movement[:, 12:]
print("joint_movement:", joint_movement.shape)

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

obs_state = []
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
    for j in range(2459):  # Run exactly 2459 frames
        if not viewer.is_running():  # Check if the viewer has been closed manually
            break
        # implement the joint angle data
        joint_angle = np.deg2rad(joint_movement[j])
        data.ctrl[:] = joint_angle
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
        
        # Record contact data
        for i in range(data.ncon):
                contact = data.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2
                # Check if the contact involves a leg geom and the floor
                for leg_index, leg_id in enumerate(leg_ids):
                    if (geom1 == leg_id and geom2 == floor_id) or (geom1 == floor_id and geom2 == leg_id):
                        contact_matrix[j, leg_index] = 1  # Mark contact

        # record the initial position
        if j == 0:
            initail_pos = data.qpos.copy()
            initail_pos = initail_pos[:]
            print("initail_pos:", initail_pos.shape)
            print("initail_pos:", initail_pos)

# record observation state and action
obs_states = np.array([obs_state]) # [1, 2459, 48] only joint angles and velocities, [1, 2459, 61] w/ torso
# simplify the states: no Sup and CTr
columns = list(range(0, 7)) + [9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30] + \
                    list(range(31, 37)) + [39, 40, 43, 44, 47, 48, 51, 52, 55, 56, 59, 60]
obs_states = obs_states[:, :, columns]
print("expert_demo:", obs_states.shape)
# np.save("Sim-StickInsect-obs.npy", obs_states)
actions = np.array(np.deg2rad(joint_movement))
print("actions:", actions.shape)
# np.save("Sim-StickInsect-act.npy", actions)
contact_matrix = np.array(contact_matrix) # [2459, 6]
print("contact_matrix:", contact_matrix.shape)
# pd.DataFrame(contact_matrix).to_csv("contact_matrix.csv", header=["LF", "LM", "LH", "RF", "RM", "RH"], index=None)


'''plotting the gait phase'''
plot_gait_phase = True
if plot_gait_phase:
    plt.figure(figsize=(7, 6))
    labels = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
    for leg in range(contact_matrix.shape[1]):
        plt.fill_between(range(contact_matrix.shape[0]), 
                        leg * 1.5, leg * 1.5 + 1, 
                        where=contact_matrix[:, leg] == 1, 
                        color='black', step='mid')
    plt.yticks([leg * 1.5 + 0.5 for leg in range(6)], ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']) 
    plt.gca().invert_yaxis()
    plt.xlabel('Time Step')
    plt.title('Gait Phase Plot')
    # plt.show()
    plt.savefig("gait_mujoco_kp300kv100.png")


'''subplot the obs and act data'''
sub_plot_obs_act = False
if sub_plot_obs_act:
    idx_j = 0 # 0--23 joint angles
    idx_v= 24 # 24--47 joint velocities
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    axs[0].plot(obs_states[0, :, idx_j+7], label="obs_states", color="blue")
    axs[0].set_title("joint angles_obs_states")
    axs[1].plot(actions[0, :, idx_j], label="actions", color="red")
    axs[1].set_title("joint angles_actions")
    axs[2].plot(obs_states[0, :, idx_v+13], label="obs_states", color="blue")
    axs[2].set_title("joint velocities_obs_states")
    axs[3].plot(actions[0, :, idx_v], label="actions", color="red")
    axs[3].set_title("joint velocities_actions")
    plt.savefig("obs_act_plot.png")


'''record the forces data'''
collect_forces_data = False
if collect_forces_data:
    contact_forces = np.array(contact_forces) # [2459, 6]
    print("contact_forces:", contact_forces.shape)
    forces_save_path = os.path.join("expert_data_builder/stick_insect", animal, "Animal12_110415_00_22_contactforce.csv")
    pd.DataFrame(contact_forces).to_csv(forces_save_path, header=["LF_foot", "LM_foot", "LH_foot", 
                                                                "RF_foot", "RM_foot", "RH_foot"], index=None)
    pd.DataFrame(contact_forces).to_csv(forces_save_path, header=["LF_sup", "LM_sup", "LH_sup", "RF_sup", "RM_sup", "RH_sup",
                                                                        "LF_CTr", "LM_CTr", "LH_CTr", "RF_CTr", "RM_CTr", "RH_CTr",
                                                                        "LF_ThC", "LM_ThC", "LH_ThC", "RF_ThC", "RM_ThC", "RH_ThC",
                                                                        "LF_FTi", "LM_FTi", "LH_FTi", "RF_FTi", "RM_FTi", "RH_FTi"], index=None)