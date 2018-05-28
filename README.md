# New GYM environments!
This repository contains the new gym environments that I will be using in my research.
Currently the repository supports the following environments:

**1. Random Walk:**

   19-state random walk described in the sutton and barto's book of reinforcement learning.
   
## Requirements
This package requires,
* python 3.6
* gym 0.10.5
* numpy 1.14.3

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
