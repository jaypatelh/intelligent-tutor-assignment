# Author: Jay H Patel (jayhp9@stanford.edu)
# Date created: 08/18/2017
# Python version: 3

import abc

import numpy as np
from scipy.special import expit

"""
You will be implementing the following methods in this file:
  AdvancedStudentConfig.explicit_difficulty_based_reward (1-2 lines)
  OKStudentConfig.explicit_difficulty_based_reward (1-2 lines)
  BeginnerStudentConfig.explicit_difficulty_based_reward (1-2 lines)
  OKToAdvStudentConfig.explicit_difficulty_based_reward (8-10 lines)
"""

class AdvancedStudentConfig:
    def __init__(self):
        pass

    def explicit_difficulty_based_reward(self, difficulty):
        """Returns the reward an advanced student would get on the given difficulty level.

        Define a dictionary containing the reward of either 0, 0.5 or 1 for each of the 5 difficulty levels.
        An advanced student is most appropriately challenged when given a difficulty level 3 or 4
        problem. Difficulty levels 0, 1 and 2 might be too easy, so we give those levels a reward of 0.
        For difficulty level 3, we give it a reward of 0.5, and for difficulty level 4, we
        give it a reward of 1.

        Our model will learn higher Q values for levels 3 and 4.

        Parameters
        ----------
        difficulty: int
            Difficulty level of a problem presented to the advanced student. Number in range [0, 4]

        Returns
        -------
        int
            The reward an advanced student would get for the given difficulty level.
        """
        # ===== YOUR CODE STARTS HERE (1-2 LINES) =====
        rewards = {
            0: 0,
            1: 0,
            2: 0,
            3: 0.5,
            4: 1,
        }
        # ===== YOUR CODE ENDS HERE =====
        return rewards[difficulty]

    def __str__(self):
        return "advanced"

class OKStudentConfig:
    def __init__(self):
        pass

    def explicit_difficulty_based_reward(self, difficulty):
        """Returns the reward an OK student would get on the given difficulty level.

        Define a dictionary containing the reward of either 0, 0.5 or 1 for each of the 5 difficulty levels.
        An OK student is most appropriately challenged when given a difficulty level 2 or 3
        problem. Difficulty levels 0 and 1 are too easy, and difficulty level 4 is too hard, so we give those 
        levels a reward of 0. For difficulty level 2, we give it a reward of 0.5, and for difficulty level 3, we
        give it a reward of 1.

        Our model will learn higher Q values for levels 2 and 3.

        Parameters
        ----------
        difficulty: int
            Difficulty level of a problem presented to the OK student. Number in range [0, 4]

        Returns
        -------
        int
            The reward an OK student would get for the given difficulty level.
        """
        # ===== YOUR CODE STARTS HERE (1-2 LINES) =====
        rewards = {
            0: 0,
            1: 0,
            2: 0.5,
            3: 1,
            4: 0,
        }
        # ===== YOUR CODE ENDS HERE =====
        return rewards[difficulty]

    def __str__(self):
        return "OK"

class BeginnerStudentConfig:
    def __init__(self):
        pass

    def explicit_difficulty_based_reward(self, difficulty):
        """Returns the reward a beginner student would get on the given difficulty level.

        Define a dictionary containing the reward of either 0, 0.5 or 1 for each of the 5 difficulty levels.
        A beginner student is most appropriately challenged when given a difficulty level 0 or 1
        problem. Difficulty levels 2, 3 and 4 are too hard, so we give those levels a reward of 0.
        For difficulty level 0, we give it a reward of 0.5, and for difficulty level 1, we
        give it a reward of 1.

        Our model will learn higher Q values for levels 0 and 1.

        Parameters
        ----------
        difficulty: int
            Difficulty level of a problem presented to the beginner student. Number in range [0, 4]

        Returns
        -------
        int
            The reward an beginner student would get for the given difficulty level.
        """
        # ===== YOUR CODE STARTS HERE (1-2 LINES) =====
        rewards = {
            0: 0.5,
            1: 1,
            2: 0,
            3: 0,
            4: 0,
        }
        # ===== YOUR CODE ENDS HERE =====
        return rewards[difficulty]

    def __str__(self):
        return "beginner"

class OKToAdvStudentConfig:
    def __init__(self):
        self.qns = 0
        self.improvement_threshold = 10

    def explicit_difficulty_based_reward(self, difficulty):
        """Returns the reward of a gradually improving student on the given difficulty level.

        Define a dictionary containing the reward of either 0, 0.5 or 1 for each of the 5 difficulty levels.
        
        At first, the student will be most appropriately challenged at levels 2 and 3. After
        self.improvement_threshold questions, the student will improve to be most appropriately challenged
        at levels 3 and 4. Like the above functions, define different dictionaries for before and after
        the improvement threshold. You can increment self.qns to track the number of questions asked so far.

        Parameters
        ----------
        difficulty: int
            Difficulty level of a problem presented to the student. Number in range [0, 4]

        Returns
        -------
        int
            The reward the student would get for the given difficulty level.
        """
        # ===== YOUR CODE STARTS HERE (8-10 LINES) =====
        rewards = None
        if self.qns >= self.improvement_threshold:
            # now the student has improved so we move up the challenge level
            rewards = {
                0: 0,
                1: 0,
                2: 0,
                3: 0.5,
                4: 1,
            }
        else:
            rewards = {
                0: 0,
                1: 0,
                2: 0.5,
                3: 1,
                4: 0,
            }
        self.qns += 1
        # ===== YOUR CODE ENDS HERE =====
        return rewards[difficulty]

    def __str__(self):
        return "OKToAdvAfter10"

class Student:
    def __init__(self, config):
        self.config = config

    def simulate_problem(self, state):
        return self.config.explicit_difficulty_based_reward(state)
