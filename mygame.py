import json
import random

import numpy as np

json, random


class MyGame():
    def __init__(self, map_size: str = "4x4"):
        self.map = []
        self.map_size = map_size
        self.map_size_list = map_size.split("x")
        self.action_map = {
            "w": 0,
            "a": 1,
            "s": 2,
            "d": 3
        }
        self.action = 4
        self.state = 0
        self.reward = 0
        self.done = False
        self.create_map()
        self.player_position = [0, 0]
        self.last_player_position = [0, 0]
        self.distance = 0
        self.last_distance = 0
        self.eps = 0
        self.last_pos = ""

    def create_map(self):
        map_x, map_y = self.map_size_list
        self.map = []
        for i in range(int(map_y)):
            self.map.append([])
            for j in range(int(map_x)):
                self.map[i].append("S")
        self.map[0][0] = "A"
        self.map[int(map_y) - 1][int(map_x) - 1] = "B"

    def print_map(self):
        for i in range(len(self.map)):
            print(self.map[i])

    def clear_console(self):
        print("\n" * 150)

    def render(self):
        player = {
            "y": self.player_position[0],
            "x": self.player_position[1]
        }
        last_player = {
            "y": self.last_player_position[0],
            "x": self.last_player_position[1]
        }
        dist_players = self.get_distance(player["x"], player["y"], last_player["x"], last_player["y"])
        if dist_players > 0:
            self.map[last_player["x"]][last_player["y"]] = self.last_pos
            self.last_player_position = [self.player_position[0], self.player_position[1]]

        self.last_pos = self.map[player["x"]][player["y"]]
        self.map[player["x"]][player["y"]] = "P"
        # self.clear_console()
        print(
            f"Distance: {self.distance}, Reward: {self.reward}, Done: {self.done}, Eps: {self.eps}, Pos: {self.player_position}")
        self.print_map()
        # time.sleep(0.3)

    def get_distance(self, x1, y1, x2, y2):
        return np.sqrt(np.sum(np.square(np.array([x1, y1]) - np.array([x2, y2]))))

    def reset(self):
        self.player_position = [np.random.randint(0, int(self.map_size_list[0])), 0]
        self.state = np.random.randint(0, int(self.map_size_list[0]))
        self.reward = 0
        self.done = False
        self.eps = 0
        self.create_map()
        return self.state

    def step(self, action):
        self.reward = 0
        self.done = False
        self.move(action)
        self.last_pos = self.map[self.player_position[0]][self.player_position[1]]

        self.goal = [int(self.map_size_list[0]) - 1, int(self.map_size_list[1]) - 1]
        goal_pos = [int(self.map_size_list[0]) - 1, int(self.map_size_list[1]) - 1]
        self.distance = np.sqrt(np.sum(np.square(np.array(goal_pos) - np.array(self.player_position))))
        if self.distance < self.last_distance:
            self.reward = 100
        else:
            self.reward = -90
        self.last_distance = self.distance
        self.eps = self.eps + 1
        if self.eps > 200:
            self.done = True
            self.reward = -100
        return self.state, self.reward, self.done, None

    def move(self, action):
        if action == 0:
            if self.player_position[0] > 0:
                self.player_position[0] -= 1
        elif action == 1:
            if self.player_position[1] < int(self.map_size_list[1]) - 1:
                self.player_position[1] += 1
        elif action == 2:
            if self.player_position[0] < int(self.map_size_list[0]) - 1:
                self.player_position[0] += 1
        elif action == 3:
            if self.player_position[1] > 0:
                self.player_position[1] -= 1
        else:
            raise ValueError("Invalid action")
        if self.map[self.player_position[0]][self.player_position[1]] == "B":
            self.done = True
        else:
            self.done = False

        x, y = self.player_position
        self.state = ((x + 1) * (y + 1)) - 1


g = MyGame()
g.reset()
done = False
# while not done:
#     action = np.random.randint(0, 3)
#     state, reward, done, _ = g.step(action)
#     # g.render()
#
#     # time.sleep(0.3)
#
# qtable = np.zeros((int(g.map_size_list[0]) * int(g.map_size_list[1]), 4))
# print(qtable)
# num_episodes = 50_000
#
# epsilon = 1.0  # Exploration rate
# max_epsilon = 1.0  # Exploration probability at start
# min_epsilon = 0.01  # Minimum exploration probability
# decay_rate = 0.005
# total_episodes = 15000  # Total episodes
# learning_rate = 0.01  # Learning rate
# max_steps = 99  # Max steps per episode
# gamma = 0.99  # Discounting rate
#
# state = g.reset()
# done = False
# reward = 0
#
# for episode in range(num_episodes):
#     state = g.reset()
#     done = False
#     reward = 0
#     while not done:
#         exp_exp_tradeoff = random.uniform(0, 1)
#
#         if exp_exp_tradeoff > epsilon:
#             action = np.argmax(qtable[state])
#         else:
#             action = np.random.randint(0, 3)
#
#         new_state, reward, done, _ = g.step(action)
#         if new_state > 15:
#             new_state = 14
#         qtable[state][action] = qtable[state][action] + learning_rate * (
#                 reward + gamma * np.max(qtable[new_state]) - qtable[state][action])
#         state = new_state
#         epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
#     if episode % 1000 == 0:
#         print(f"Episode: {episode}, Exploration: {epsilon}")
#
#         with open('qtable_my.json', 'w') as fp:
#             json.dump(qtable.tolist(), fp)
# print(qtable)
