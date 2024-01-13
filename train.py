import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from envs import *
import numpy as np
import pandas as pd
import mujoco_py
from sklearn.preprocessing import MinMaxScaler
import yaml
import datetime
import dateutil.tz
from algorithms.maxent_irl import *
import gym

n_states = 144 # position - 12, velocity - 12
n_actions = 12
one_feature = 12 # number of state per one feature
q_table = np.zeros((n_states, n_actions)) # (144, 12)
feature_matrix = np.eye((n_states)) # (144, 144)

gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05

np.random.seed(1)

def dataset_normalization(dataset):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(dataset)
    ds_scaled = scaler.transform(dataset)
    # denormalize the dataset
    # ds_rescaled = scaler.inverse_transform(ds_scaled)
    return scaler, ds_scaled

def fold_configure(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)  

def idx_demo(env, one_feature):
    env_low = env.observation_space.low     
    env_high = env.observation_space.high   
    env_distance = (env_high - env_low) / one_feature  

    raw_demo = np.load(file="expert_demo.npy")
    demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))

    for x in range(len(raw_demo)):
        for y in range(len(raw_demo[0])):
            position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
            velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
            state_idx = position_idx + velocity_idx * one_feature

            demonstrations[x][y][0] = state_idx
            demonstrations[x][y][1] = raw_demo[x][y][2] 
            
    return demonstrations

def idx_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high 
    env_distance = (env_high - env_low) / one_feature 
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature
    return state_idx

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)


def main():
    # Load joint angle data from the CSV file
    csv_file_path = 'expert_data_builder/demo_dataset.csv'  
    dataset = pd.read_csv(csv_file_path, header=0, 
                          usecols=[1,2,3,4,5,6,7,8,9,10,11,12]).to_numpy()
    #dataset = dataset[5200:6200, :]
    # open config file
    with open("configs/irl.yml", "r") as f:
        config_data = yaml.safe_load(f)

    #  Set up simulation without rendering
    model_name = config_data.get("model")
    model_path = 'envs/assets/' + model_name + '.xml'
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    # viewer = mujoco_py.MjViewer(sim)

    # Get the state trajectories [12,len(dataset).12]
    trajectories = []
    for i in range(len(dataset)):
        joint_angle = np.deg2rad(dataset[i])
        sim.data.ctrl[:] = joint_angle
        sim.step()
        # qpos: joint positions
        state_pos = sim.get_state().qpos.copy()
        state_vel = sim.get_state().qvel.copy()
        action = sim.data.ctrl.copy()

    trajectories = np.array(trajectories)
    pd.DataFrame(trajectories).to_csv("state_trajectories.csv", 
                                                                                        header=None, index=None)

    env = gym.make('CricketEnv2D-v0')
    demonstrations = idx_demo(env, one_feature)

    expert = expert_feature_expectations(feature_matrix, demonstrations)
    learner_feature_expectations = np.zeros(n_states)

    theta = -(np.random.uniform(size=(n_states,)))

    episodes, scores = [], []

    for episode in range(30000):
        state = env.reset()
        score = 0

        if (episode != 0 and episode == 10000) or (episode > 10000 and episode % 5000 == 0):
            learner = learner_feature_expectations / episode
            maxent_irl(expert, learner, theta, theta_learning_rate)
                
        while True:
            state_idx = idx_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)
            
            irl_reward = get_reward(feature_matrix, theta, n_states, state_idx)
            next_state_idx = idx_state(env, next_state)
            update_q_table(state_idx, action, irl_reward, next_state_idx)
            
            learner_feature_expectations += feature_matrix[int(state_idx)]

            score += reward
            state = next_state
            
            if done:
                scores.append(score)
                episodes.append(episode)
                break

        if episode % 1000 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./learning_curves/maxent_30000.png")
            np.save("./results/maxent_q_table", arr=q_table)

if __name__ == '__main__':
    main()