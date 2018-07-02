# New GYM environments!
This repository contains the new gym environments that I will be using in my research.
Currently the repository supports the following environments:

**1. Random Walk:** 19-state random walk described in the sutton and barto's book of reinforcement learning. **RandomWalk-v0** is the environment id.

**2. Grid World:** There are two maps in the grid world environment, a 4x4 version and a 8x8 version. The start state is (0,0), the goal state is (3,3) (in case of 4x4 grid) and (7,7) (in case of 8x8 grid). Each state has 4 possible actions west:0, east:2, south:1, north:3. The goal state has a reward of +1. There is a slippery element involved. The agents takes the action you ask it to take only 80% of the times and for the rest of the times, it takes actions perpendicular to the direction you ask it to take with a proabaility of 10% for either of the perpendicular actions. **gridWorld-v0 and gridWorld-v1** are the environment ids for 4x4 and 8x8 versions respectively.

**3. POMDP Grid World:** These grid worlds are the same except that the agent state is different from the actual state. The agent is a tuple of 4 boolean numbers, can be thought of a set of 4 sensor readings with each sensor answering the following questions:
  * does state exist towards top?
  * does state exist towards right?
  * does state exist towards bottom?
  * does state exist towards left?

**pomdpGridWorld-v0 and pomdpGridWorld-v1** are the environment ids for 4x4 and 8x8 versions respectively.

**4. Parr and Russell Grid World:** This is a simple 4x3 grid world with two termination states,
  * A positive terminal state with a reward of +1
  * A negative terminal state with a reward of -1
The observation is a tuple of two elements with each element answering the following question,
  * Is there an obstacle towards my left?
  * Is there an obstacle towards my right?
At each timestep the agent gets a reward of -0.04. Agent can take any of the 4 actions - West, South, East and North.

**parr_russell-v0** is the environment id.
   
## Requirements
This package requires,
* python >= 3.6
* gym >= 0.10.5
* numpy >= 1.14.3

## Installation
* clone the repository
* add the repository to the PYTHONPATH or you can use the following piece of code in the main code
  ```python
  import sys
  sys.path.append('add local path to this repository')
  import random_walk # you can import the environment that you want to import
  ```
* You can use this environment just like any other gym environment! For example,
  ```python
  import gym
  env = gym.make('RandomWalk-v0') # 19-state random walk environment is created.
  ```
  creates a new 19-state RandomWalk environment.
