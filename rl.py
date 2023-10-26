import torch
from rl.algorithms.sac import SAC


if __name__ == "__main__":
    """Soft Actor-Critic RL implementation based off of OpenAI's Spinning Up Guide"""

    import gym
    import matplotlib.pyplot as plt
    from gym.wrappers import Monitor
    from envs.CricketEnv import CricketEnv

    torch.set_num_threads(torch.get_num_threads())
    
    env = CricketEnv(max_timesteps=500).unwrapped
    # Crucial! This gets rid of the time limit so it doesn't stop at 1000 steps every time
    sac = SAC(
        env,
        hidden_sizes=(256, 256),
        update_freq=64,
        num_update=64,
        update_threshold=4096,
        batch_size=128,
        alpha=0.5,
        device="cuda:0",
    )
    training_stats = sac.train(steps=500_000, render_freq=50_000, max_steps=10000)

    while input("Type 'quit' to quit: ") != "quit":
        sac.test(render=True)

    plt.figure(figsize=(12, 8))
    for idx, (title, data) in enumerate(training_stats.items(), start=1):
        plt.subplot(1, 4, idx)
        plt.plot(data)
        plt.title(title)

    plt.savefig("sac-2000k-ant-v3.png")
