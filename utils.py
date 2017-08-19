# Author: Jay H Patel (jayhp9@stanford.edu)
# Date created: 08/18/2017
# Python version: 3

from abc import ABC, abstractmethod

import numpy as np
import random

from mdp import State

"""
You will be implementing the following methods in this file:
  modified_greedy_action (4-6 lines)
"""

def modified_greedy_action(eps, Q, state):
    """Choose the greedy action to take from the given state, except for a higher action
    sometimes. Here, the action represents the difficulty of the problem, so a higher action
    refers to a harder problem.

    A greedy action is the action with the highest Q value from the given state.

    Parameters
    ----------
    eps: float 
        The probability of choosing a harder action instead of the greedy action.
    Q: np.array
        An array of shape [env.nS x env.nA] representing state, action values
    state: int
        The current state, from which the action should be chosen. Number in range [0, 4]
      
    Returns
    -------
        int
            The action to take. Number in range [0, 4]
    """
    # ===== YOUR CODE STARTS HERE (4-6 LINES) =====
    # greedy action
    action = np.argmax(Q[state])

    # with some probability, take the more challenging action
    flip = random.random()
    if flip < eps and action < len(Q[state]) - 1:
        action += 1
    # ===== YOUR CODE ENDS HERE =====
    return action

def print_Q(Q):
    pretty = ""
    for state in range(len(Q)):
        pretty += "State: " + str(state) + "\n"
        for action in range(len(Q[0])):
            pretty += "\t" + "Action: " + str(action) + ": " + str(Q[state][action]) + "\n"
        pretty += "\n"

    print(pretty)
