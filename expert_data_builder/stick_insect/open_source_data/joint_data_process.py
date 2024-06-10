import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data_path = 'sup_process_source.csv'
dataset = pd.read_csv(data_path, header=[0], index_col=None).to_numpy()
print(dataset.shape)

fem = dataset[:, :6]
cox_y = dataset[:, 6:]
print(fem.shape, cox_y.shape)

lev_dep = fem + cox_y
print(lev_dep.shape)

save_path = 'sup_data.csv'
pd.DataFrame(lev_dep).to_csv(save_path, header=None, index=None)