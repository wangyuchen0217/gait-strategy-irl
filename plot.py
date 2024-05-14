# plot the data from expert_data_builder/movement/c21/PIC0680_Heading_direction.csv

import matplotlib.pyplot as plt
import pandas as pd

# read the data
direction_path = 'expert_data_builder/movement/c21/PIC0680_Heading_direction.csv'
direction = pd.read_csv(direction_path, index_col=0, header=0).to_numpy()

# set the initial direction to 0
direction = direction - direction[0]

# plot the data
plt.figure(figsize=(15, 5))
plt.plot(direction)
plt.xlabel('Frame')
plt.ylabel('Direction')   
plt.title('c21_0680_direction')
plt.grid()
plt.savefig('c21_0680_direction.png')
