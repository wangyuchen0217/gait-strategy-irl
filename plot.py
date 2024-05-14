# plot the data from expert_data_builder/movement/c21/PIC0680_Heading_direction.csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# calculate the trajectory
direction_rad = direction * np.pi / 180
direction_x = np.cos(direction_rad)
direction_y = np.sin(direction_rad)
trajectory = []
x = 0
y = 0
for i in range(len(direction)):
    x += 1 * direction_x[i]
    y += 1 * direction_y[i]
    trajectory.append([x, y])

# plot the trajectory
plt.figure()
trajectory = pd.DataFrame(trajectory)
plt.plot(trajectory[0], trajectory[1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('c21_0680_trajectory')
plt.grid()
plt.show()