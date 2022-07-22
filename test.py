import json

import gym
import numpy as np

with open('qtable.json', 'r') as f:
    qtable = json.load(f)
env = gym.make("FrozenLake-v0")

state = env.reset()
done = False
reward = 0
while not done:
    action = np.argmax(qtable[state])
    state, reward, done, _ = env.step(action)
    print(reward, state)
    env.render()
print(f"Reward: {reward}")
