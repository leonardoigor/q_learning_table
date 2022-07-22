import gym

env = gym.make("FrozenLake-v0", map_name="4x4")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
print(action_space_size, state_space_size)
