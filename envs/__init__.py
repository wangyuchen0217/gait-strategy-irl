#!/usr/bin/env python
from gym.envs.registration import register

register(
    id='AntFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Ant-v2'
    )
)

# A modified ant which flips over less and learns faster via TRPO
register(
    id='CustomAnt-v0', 
    entry_point='envs.ant_env:CustomAntEnv',
    kwargs={'gear': 30, 'disabled': False})

register(
    id='DisabledAnt-v0', 
    entry_point='envs.ant_env:CustomAntEnv',
    kwargs={'gear': 30, 'disabled': True})
