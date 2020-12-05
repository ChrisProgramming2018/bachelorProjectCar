import os
import gym
import random
import torch
import numpy as np
from collections import deque


env = gym.make('Freeway-v0')
state = env.reset()
score = 0 
t  = 0
while True:
    t += 1
    print(t)
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    score += reward
    env.render()
    if done:
        print(score)
        print("Evaluate policy on {} t".format(t))   
        break 
