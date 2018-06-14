from gym.envs.registration import register

register(
    id='pomdpGridWorld-v0',
    entry_point='pomdp_grid_world.envs:gridWorld',
    kwargs={'map_name' : '4x4'},
    max_episode_steps=100, 
)

register(
    id='pomdpGridWorld-v1',
    entry_point='pomdp_grid_world.envs:gridWorld',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
)