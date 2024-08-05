#!/usr/bin/env python
from gym.envs.registration import register

register(
    id='AntFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(env_name='Ant-v2')
)

register(
    id='CricketEnv-v0',
    entry_point='envs.CricketEnv3D:CricketEnv'
)

register(
    id='CricketEnv2D-v0',
    entry_point='envs.CricketEnv2D:CricketEnv2D'
)


from gymnasium.envs.registration import register

register(
     id='StickInsect-v0',
     entry_point='envs.StickInsectEnv:StickInsectEnv',
     max_episode_steps=3000,
)

register(
     id='StickInsect-v0-disc',
     entry_point='envs.StickInsectEnv_disc:StickInsectEnv',
     max_episode_steps=3000,
)

register(
     id='Sim-v0',
     entry_point='envs.SimEnv:StickInsectEnv',
     max_episode_steps=3000,
)