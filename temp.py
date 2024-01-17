import numpy as np

fearure_matrix = np.eye(4)
# random numpy array with all integers
trajectory = np.random.randint(10, size=(3, 2, 4))
print(trajectory)

a = np.random.randint(10, size=(2, 5))
print(a)
print(a.shape[1])