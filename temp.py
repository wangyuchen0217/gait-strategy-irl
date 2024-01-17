import numpy as np

# random numpy array with all integers
trajecroty = np.random.randint(10, size=(5, 2))
print(trajecroty)

print(trajecroty[0,0])

feature_matrix = np.random.randint(10, size=(4, 1))
print(feature_matrix.shape[0])