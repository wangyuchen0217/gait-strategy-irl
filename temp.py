import numpy as np

# random numpy array with all integers
trajecroty = np.random.randint(10, size=(3, 2, 4))
print(trajecroty)

for state, action in trajecroty:
    print(state)
    print(action)
    print()

#enumerate
for i, (state, action) in enumerate(trajecroty):
    print(i)
    print(state)
    print(action)
    print()