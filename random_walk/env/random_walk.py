import gym
from gym import spaces
from gym.utils import seeding

class RandomWalk(gym.Env):
    """
    This environment imports random walk environment.
    
    Two actions are possible:
    0: Moves up the chain (Right)
    1: Moves down the chain (Left)
    
    There are two terminal states location at both the extremes
    of the chain of states.
    1. The extreme right of the walk has a reward of +1
    2. The extreme left of the walk has a reward of 0
    
    The agent starts in a state that is located in between both
    these terminal states.
    """
    def __init__(self, n=19, slip=0.2, small=0, large=1):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 10  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        assert self.action_space.contains(action)
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if state!= 0 and state!= 20:
            if action:
                self.state -= 1
            else:
                self.state += 1
        else:
            if state == 1:
                reward = self.small
            elif state == 1:
                reward = self.large
            done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = 10
        return self.state