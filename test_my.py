import json
import random

import numpy as np

from mygame import MyGame

with open('qtable_my.json', 'r') as f:
    qtable = json.load(f)

env = MyGame()
state = env.reset()
done = False
reward = 0
epsilon = .3
while not done:
    exp_exp_tradeoff = random.uniform(0, 1)
    if exp_exp_tradeoff > epsilon:
        action = np.argmax(qtable[state])
    else:
        action = np.random.randint(0, 3)
    action = np.argmax(qtable[state])
    state, reward_, done, _ = env.step(action)
    print(f"action: {action}")
    env.render()
    reward += reward_
print(f"Reward: {reward}")
