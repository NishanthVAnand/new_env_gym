# New GYM environments!
This repository contains the new gym environments that I will be using in my research.
Currently the repository supports the following environments:

**1. Random Walk:** 19-state random walk described in the sutton and barto's book of reinforcement learning.

**2. Grid World:** There are two maps in the grid world environment, a 4x4 version and a 8x8 version. The start state is (0,0), the goal state is (3,3) (in case of 4x4 grid) and (7,7) (in case of 8x8 grid). Each state has 4 possible actions west:0, east:2, south:1, north:3. The goal state has a reward of +1. There is a slippery element involved. The agents takes the action you ask it to take only 80% of the times and for the rest of the times, it takes actions perpendicular to the direction you ask it to take with a proabaility of 10% for either of the perpendicular actions.
   
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
