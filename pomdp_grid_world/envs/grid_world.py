import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete
from gym import Env, spaces
from gym.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],
}

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Tuple((spaces.Discrete(2),spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return (0, 1, 1, 0)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d, pomdp_s = transitions[i]
        self.s = s
        self.lastaction=a
        return (pomdp_s, r, d, {"prob" : p})

class gridWorld(DiscreteEnv):
    """
    This class implements a simple grid world. There are two variants
    1. 4x4 version
    2. 8x8 version
    - The start state is (0,0) and the goal state is (3,3) in case of 4x4 grid
    and (7,7) in case of 8x8 grid
    - Each state has 4 possible actions
        left:0
        right:2
        down:1
        up:3
    - The goal state has a reward of +1
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            """
            Given a row and a column id, represent them in the state number.
            - states go from 0 to (ncol*nrow - 1)
            """
            return row*ncol + col

        def to_pomdp_s(row, col):
            """
            Given a row and a column id, represent them in the pomdp state format.
            - pomdp state format is a tuple of 4 booleans where each boolean answers
            the following question:
                1. does state exist towards top?
                2. does state exist towards right?
                3. does state exist towards bottom?
                4. does state exist towards left?
            - states go from 0 to (ncol*nrow - 1)
            """
            return (int(row-1 >= 0), int(col+1 < self.ncol), int(row+1 < self.nrow), int(col-1 >= 0))
        
        def inc(row, col, a):
            """
            defines what will be the next state (row, col) format after the application
            of action a in the current state.
            """
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                agent_s = to_pomdp_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True, agent_s))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                agent_newstate = to_pomdp_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                if b == a:
                                    li.append((0.8, newstate, rew, done, agent_newstate))
                                else:
                                    li.append((0.1, newstate, rew, done, agent_newstate))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            agent_newstate = to_pomdp_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done, agent_newstate))

        super(gridWorld, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile