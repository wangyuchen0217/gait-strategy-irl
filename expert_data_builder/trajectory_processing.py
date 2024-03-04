import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco_py
import yaml
import json

cricket_number = 'c21'
video_number = '0680'
# read the joint movement data
csv_file_path = os.path.join("expert_data_builder/velocity_data", cricket_number, 
                                            f"{video_number}_Velocity_Smooth.csv")
vel = pd.read_csv(csv_file_path, header=None, usecols=[1,2]).to_numpy() # vel.x and vel.y
# scale velocity: mm/s
vel = vel * 0.224077
# frequency: 119.88(120)Hz
# trajectory calculation
traj_x = [0]; x=0
traj_y = [0]; y=0
for i in range(1, len(vel)):
    x = x + vel[i][0]*1/119.88/1000
    y = y + vel[i][1]*1/119.88/1000
    traj_x.append(x)
    traj_y.append(y)

plt.figure(figsize=(6, 6))
plt.plot(traj_x[:], traj_y[:])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('X Corodinate [m]',fontsize=8)
plt.ylabel('Y Corodinate [m]',fontsize=8)
plt.xlim(-2.5, 2.5)
plt.ylim(-4.8, 0.2)
plt.grid(True)
plt.savefig(f"expert_data_builder/{cricket_number}_{video_number}_trajectory.png")