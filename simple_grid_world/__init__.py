from gym.envs.registration import register

register(
    id='gridWorld-v0',
    entry_point='simple_grid_world.envs:gridWorld',
    kwargs={'map_name' : '4x4'},
    max_episode_steps=100,
    reward_threshold=0.78, 
)

register(
    id='gridWorld-v1',
    entry_point='simple_grid_world.envs:gridWorld',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99,
)