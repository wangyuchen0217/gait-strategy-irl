import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据
data = pd.DataFrame({
    "Action": ["0","1","2","3","4","5"] * 3,
    "Policy": ["vel-dir"]*6 + ["vel-acc"]*6 + ["antenna"]*6,
    "MHD": [0.500,0.363,0.500,0.500,0.500,0.500,
            0.500,0.454,0.500,0.500,0.500,0.499,
            0.500,0.500,0.461,0.488,0.484,0.484],
    "SWD": [0.003,0.107,0.180,0.135,0.188,0.313,
            0.001,0.059,0.234,0.128,0.276,0.381,
            0.010,0.110,0.150,0.112,0.157,0.253]
})

fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)
sns.barplot(data=data, x="Action", y="MHD", hue="Policy", ax=ax[0])
sns.barplot(data=data, x="Action", y="SWD", hue="Policy", ax=ax[1])
ax[0].set_title("Mean Hausdorff Distance (MHD)")
ax[1].set_title("Sliced Wasserstein Distance (SWD)")
plt.tight_layout()
plt.show()
