# Author: Jay H Patel (jayhp9@stanford.edu)
# Date created: 08/18/2017
# Python version: 3

import numpy as np
import random
import matplotlib.pyplot as plt

"""
You will be implementing the following functions in this file:
  eps_greedy_action (6-8 lines)
  collect_rewards (7-10 lines)
  compute_probabilities (14-20 lines)
"""

def eps_greedy_action(env, eps, Q, curr_state):
  """From the 'curr_state', choose the optimal action to take next based on the 'Q' values.
  With probability 'eps', choose a random action instead, to allow for some exploration.

  Get a random variable between 0 and 1. If it's less than eps, choose a random action between 0 and 4.
  Else, choose the action with the highest Q value, for the given curr_state.

  Remember, Q is a 2-D numpy array containing values of each action from each state. Actions and states 
  are 0-indexed. This means the fifth action and fifth state are both indexed at 4. For example, to find the
  value of the first action (index 0) from the fifth state (index 4), do Q[4][0].

  Parameters
  ----------
  env: gym.core.Environment
    Environment object containing attributes nS (num states), nA (num actions per state), 
    and P (transition probabilities).
  eps: float 
    Epsilon value used in the epsilon-greedy method. Reflects the probability of choosing a random action
    instead of the greedy method.
  Q: np.array
    An array of shape [env.nS x env.nA] representing state, action values
  curr_state: int
    The current state, from which the action should be chosen. Number in range [0, 4]
  
  Returns
  -------
  int
    The action to take. Number in range [0, 4]
  """

  # ===== YOUR CODE STARTS HERE (6-8 LINES) =====
  # choose action based on epsilon-greedy
  flip = random.random()
  action = None
  if flip < eps:
    # choose random action
    action = random.randint(0, env.nA-1)
  else:
    # choose greedy action
    action = np.argmax(Q[curr_state])
  # ===== YOUR CODE ENDS HERE =====
  return action

def collect_rewards(env, Q, num_episodes=1000):
  """Explore the environment for num_episodes episodes, and collect the rewards at each state
    in the rewards_per_state dictionary. In the compute_probabilities() function, we will 
    aggregate this data to compute the probability of receiving a reward in a given state.

    Recall that we have 5 states in total, designated as 0, 1, 2, 3 and 4. We have provided the 
    dictionary below with states as keys, and empty arrays as values.

    To run an episode, first reset the environment (returns the initial state), then choose the greedy
    action from that state. This gives you a reward and leads you to another state. Save the reward for the state
    and then choose a greedy action from the new state. Do this until the episode finishes (reflected by 
    'done' being True). In this assignment, every episode finishes after 10 steps. Finally return the
    filled rewards_per_state dictionary.

    Parameters
    ----------
    env: gym.core.Environment
      Environment object containing attributes nS (num states), nA (num actions per state), 
      and P (transition probabilities).
    Q: np.array
      An array of shape [env.nS x env.nA] representing state, action values
    num_episodes: int
      The number of episodes to collect rewards for
  
    Returns
    -------
    dict
      A dictionary with state as key, and an array of collected rewards as the value.
  """
  rewards_per_state = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
  }

  # ===== YOUR CODE STARTS HERE (7-10 LINES) =====
  for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
      action = np.argmax(Q[state])
      state, reward, done = env.step(action)
      rewards_per_state[state].append(reward)
  # ===== YOUR CODE ENDS HERE =====

  return rewards_per_state

def compute_probabilities(rewards_per_state):
  """For each state, compute probabilities of receiving each reward by counting occurrences in
  the rewards_per_state dict (returned by collect_rewards function).

  We "smooth" our counts by adding one to all of them. This is useful to ensure that a reward with zero occurrence
  yields a non-zero probability. The intuition is that just because we didn't collect that reward during our
  exploration episodes doesn't mean we won't ever see it in future episodes.

  For example, if the collected rewards for state 2 were [0.5, 0.5, 1, 0.5, 1], then
  the probability of receiving reward 0 is (0+1)/(5+3) = 1/8, the probability of receiving 
  reward 0.5 is (3+1)/(5+3) = 4/8 = 1/2 and the probability of receiving reward 1 is (2+1)/(5+3) = 3/8.
  Notice how we added a 1 to the numerators (making the probability of reward 0 be non-zero), and
  then to normalize it properly, added 3 to the denominator.

  Rewards can be either 0, 0.5 or 1. (You will be implementing these rewards in student.py)

  For each state, count the frequencies of each reward in it's corresponding array, and divide each by the
  total number of collected rewards for that state to obtain a probability. Remember to "smooth" counts as
  explained above. Finally, store these probabilities in a dictionary where the key is the state, and 
  the value is another dictionary. The key in this inner dictionary is the reward and it's value is 
  the probability of that reward occurring from the state.

  Parameters
  ----------
  rewards_per_state: dict of state to array of rewards
    Rewards collected during exploration per state
  
  Returns
  -------
  dict
    A dictionary with state as key, and a dictionary as it's value. The inner dictionary has the reward
    as the key and the probability of that reward occuring as the value.
  """
  rewards_probs = dict()
  # ===== YOUR CODE STARTS HERE (14-20 LINES) =====
  for state in rewards_per_state:
    uniq, counts = np.unique(rewards_per_state[state], return_counts=True)
    total = np.sum(counts)
    
    reward_candidates = [0, 0.5, 1]
    rewards_probs[state] = dict()
    for candidate in reward_candidates:
      rewards_probs[state][candidate] = 0

    for r in reward_candidates:
      if r in uniq:
        idx = np.where(uniq==r)[0][0]
        rewards_probs[state][r] = float(counts[idx] + 1) / (total + len(reward_candidates))
      else:
        rewards_probs[state][r] = 1.0 / (total + len(reward_candidates))
  # ===== YOUR CODE ENDS HERE =====
  return rewards_probs

def reward_per_state_probs(env, Q, num_episodes=1000):
  rewards = collect_rewards(env, Q, num_episodes)
  rewards_probs = compute_probabilities(rewards)
  return rewards_probs

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.2, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  def make_update(Q, state, action, next_state, next_action, r, gamma, lr):
    if next_action is not None:
      state_action_val = Q[state][action]
      next_state_action_val = Q[next_state][next_action]
      error = r + (float(gamma) * next_state_action_val) - state_action_val
      weighted_error = float(lr) * error
      Q[state][action] += weighted_error
    else:
      error = r - Q[state][action]
      weighted_error = float(lr) * error
      Q[state][action] += weighted_error

  Q = np.ones((env.nS, env.nA))

  # for plotting
  episode_numbers = []
  running_average_rewards = []
  total_rewards = 0.0
  eps_rewards = []

  for episode in range(num_episodes):
    path = ""
    episode_reward = 0
    # start state
    state = env.reset()
    path += str(state) + " "

    visited = {}

    # for debugging
    num_actions = 0

    while True:
      # choose action using behavioral policy and act on it
      action = eps_greedy_action(env, e, Q, state)
      next_state, r, terminal = env.step(action)
      path += str(next_state) + " "

      total_rewards += r
      episode_reward += r

      # for debugging
      num_actions += 1

      if not terminal:
        # find max possible value from next_state onwards
        max_action = np.argmax(Q[next_state])

        visited[(state, action)] = (next_state, max_action, r)

        # update Q
        make_update(Q, state, action, next_state, max_action, r, gamma, lr)

        # update state to be next_state
        state = next_state
      else:
        visited[(state, action)] = (next_state, None, r)
        make_update(Q, state, action, next_state, None, r, gamma, lr)

        # WE ARE SIMULATING TERMINAL STATE SO DON'T WANT TO WIPE OUT Q VALUES LIKE THIS
        #for a in range(env.nA):
        #  Q[next_state][a] = 0.0

        break

    if episode < 1000:
      episode_numbers.append(episode)
      running_average_rewards.append(float(total_rewards) / len(episode_numbers))

    # sample a previous state-action pair from this episode and update again
    sample = random.choice(list(visited)) # list(visited) gives a list of the keys in visited in Py3
    make_update(Q, sample[0], sample[1], visited[sample][0], visited[sample][1], visited[sample][2], gamma, lr)

    # decay the epsilon rate
    e = float(decay_rate) * e

    eps_rewards.append(episode_reward)

  return Q
