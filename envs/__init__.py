#!/usr/bin/env python
from gym.envs.registration import register

register(
    id='AntFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(env_name='Ant-v2')
)

register(
    id='CricketEnv-v0',
    entry_point='envs.point_maze_env:CricketEnv',
    kwargs={'sparse_reward': False, 'direction': 1}
)