from gym.envs.registration import register

register(
    id='parr_russell-v0',
    entry_point='parr_russell.envs:gridWorld',
    kwargs={'map_name' : '4x3'}, 
)