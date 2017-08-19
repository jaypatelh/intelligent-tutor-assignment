# Author: Jay H Patel (jayhp9@stanford.edu)
# Date created: 08/18/2017
# Python version: 3

import numpy as np

from qlearning import learn_Q_QLearning, reward_per_state_probs
from rmax import rmax_learning

"""
    Nothing for you to implement in this file! Just read and enjoy.
"""

class Env:
    def __init__(self, student, initial_state, nS, nA, episode_length):
        self.student = student
        self.initial_state = initial_state
        self.state = self.initial_state
        self.nS = nS
        self.nA = nA
        self.num_steps = 0
        self.episode_length = episode_length

    def reset(self):
        self.state = self.initial_state
        self.num_steps = 0
        return self.state

    def step(self, action):
        """
        Input action can be 1, 2, 3, 4 or 5, corresponding to jumping to a state of the same number.
        """
        rew = self.student.simulate_problem(action)
        self.state = action
        self.num_steps += 1
        terminal = False
        if self.num_steps == self.episode_length:
            terminal = True
        return action, rew, terminal

class Agent:
    def __init__(self, env, algo="qlearning"):
        self.env = env
        self.Q = None
        self.reward_probabilities = None
        self.algo = algo

    def learn_q(self):
        if self.algo == "qlearning":
            self.Q = learn_Q_QLearning(self.env)
        elif self.algo == "rmax":
            self.Q = rmax_learning(self.env, m=10, R_max=1)

    def compute_probs_reward_per_state(self):
        self.reward_probabilities = reward_per_state_probs(self.env, self.Q)

    def render_episodes(self, num):
        for episode in range(num):
            state = self.env.reset()
            path = str(state) + " "
            done = False
            while not done:
                action = np.argmax(self.Q[state])
                state, reward, done = self.env.step(action)
                path += str(state) + " "
            print(path)

    def greedy_action(self, state):
        return np.argmax(self.Q[state])

class State:
    VERY_LOW_UNDERSTANDING_STATE = 0
    LOW_UNDERSTANDING_STATE = 1
    MEDIUM_UNDERSTANDING_STATE = 2
    HIGH_UNDERSTANDING_STATE = 3
    VERY_HIGH_UNDERSTANDING_STATE = 4
