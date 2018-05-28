from gym.envs.registration import register

register(
    id='RandomWalk-v0',
    entry_point='random_walk.env:RandomWalk',
    timestep_limit=100,
)