# plot the data from expert_data_builder/movement/c21/PIC0680_Heading_direction.csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''direction'''

# read the data
direction_path = 'expert_data_builder/movement/c21/PIC0680_Heading_direction.csv'
direction = pd.read_csv(direction_path, index_col=0, header=0).to_numpy()

# set the initial direction to 0
direction = direction - direction[0]

# plot the direction
plt.figure(figsize=(15, 5))
plt.plot(direction)
plt.xlabel('Frame')
plt.ylabel('Direction')   
plt.title('c21_0680_direction')
plt.grid()
plt.savefig('c21_0680_direction.png')

'''trajectory'''

# calculate the trajectory
direction = direction.flatten()
direction_rad = direction * np.pi / 180
direction_x = np.cos(direction_rad)
direction_y = - np.sin(direction_rad)
x, y = 0, 0
trajectory_x = []
trajectory_y = []
trajectory_x.append(x)
trajectory_y.append(y)
for i in range(len(direction)):
    x += 1 * direction_x[i]
    y += 1 * direction_y[i]
    trajectory_x.append(x)
    trajectory_y.append(y)

trajectory_x = np.array(trajectory_x)
trajectory_y = np.array(trajectory_y)
print("trajectory_x: ", trajectory_x.shape)
print("trajectory_y: ", trajectory_y.shape)

# plot the trajectory
plt.figure()
plt.plot(trajectory_x, trajectory_y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('c21_0680_trajectory')
plt.grid()
plt.savefig('c21_0680_trajectory.png')