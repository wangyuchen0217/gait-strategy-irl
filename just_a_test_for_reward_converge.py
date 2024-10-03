import numpy as np
import matplotlib.pyplot as plt

mean_rewards = []
th4s = []
maxs = []
for i in range(15):
    no = (i + 1)*10
    datapath = 'direc_inferred_rewards' + str(no) + '.csv'
    # calculate the mean reward
    data = np.loadtxt(datapath, delimiter=',')
    mean_reward = np.mean(data)
    print('mean reward:', mean_reward)
    mean_rewards.append(mean_reward)
    th4 = data[3]
    print('th4:', th4)
    th4s.append(th4)
    max = np.max(data)
    print('max:', max)
    maxs.append(max)


plt.figure(figsize=(10, 8))
plt.plot(maxs)
plt.show()


# means_rewards = []
# th4s = []
# maxs = []
# for i in range(10):
#     no = (i + 1)*10
#     datapath = 'direc_inferred_rewards' + str(no) + '.csv'
#     # calculate the mean reward
#     data = np.loadtxt(datapath, delimiter=',')
#     data = data.sum(axis=1)
#     mean_reward = np.mean(data)
#     print('mean reward:', mean_reward)
#     means_rewards.append(mean_reward)
#     th4 = data[80]
#     print('th4:', th4)
#     th4s.append(th4)
#     max = np.max(data)
#     print('max:', max)
#     maxs.append(max)

# plt.figure(figsize=(10, 8))
# plt.plot(th4s)
# plt.show()