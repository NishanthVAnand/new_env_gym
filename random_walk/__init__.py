from gym.envs.registration import register

register(
    id='RandomWalk-v0',
    entry_point='random_walk.envs:RandomWalk',
    timestep_limit=100,
)