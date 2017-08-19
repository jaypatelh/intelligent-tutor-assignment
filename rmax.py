import numpy as np

def rmax_learning(env, m, R_max, gamma=0.95, epsilon=0.8, num_episodes=2000, max_step=10):
    """Learn state-action values using the Rmax algorithm

    Args:
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    m: int
        Threshold of visitance
    R_max: float 
        The estimated max reward that could be obtained in the game
    epsilon: 
        accuracy paramter
    num_episodes: int 
        Number of episodes of training.
    max_step: Int
        max number of steps in each episode

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    Q = np.ones((env.nS, env.nA)) * R_max / float(1 - gamma) # our current Q values (i.e. policy)
    R = np.zeros((env.nS, env.nA)) # RSUM: real total observed rewards so far
    nSA = np.zeros((env.nS, env.nA)) # real visited counts so far
    nSASP = np.zeros((env.nS, env.nA, env.nS)) # real visited counts so far
    ########################################################
    #                   YOUR CODE HERE                     #
    ########################################################
    num_value_iterations = int(np.ceil(np.log(1.0 / (float(epsilon) * (1 - gamma))) / float(1 - gamma)))
    total_rewards = 0
    running_average_rewards = []

    # start exploring based on greedy actions from current Q
    for episode in range(num_episodes):
        # initialize start state
        state = env.reset()
        episode_reward = 0

        for step in range(max_step):
            # choose optimal action based on current Q values
            max_a = np.argmax(Q[state])

            # take the action
            nextstate, rew, done = env.step(max_a)
            episode_reward += rew

            # update counts and rewards
            R[state][max_a] += rew
            nSA[state][max_a] += 1
            nSASP[state][max_a][nextstate] += 1

            if nSA[state][max_a] == m: # time to update some Q values
                # first update model of the environment for the 'known' states
                model_R = np.zeros((env.nS, env.nA))
                model_P = np.zeros((env.nS, env.nA, env.nS)) # transition probabilities
                for s in range(env.nS):
                    for a in range(env.nA):
                        if nSA[s][a] >= m:
                            # update model of the environment for this 'known' state-action pair
                            model_R[s][a] = float(R[s][a]) / nSA[s][a]
                            for ns in range(env.nS):
                                model_P[s][a][ns] = float(nSASP[s][a][ns]) / nSA[s][a]

                # conduct value iteration for all states with >= m, using model
                for iteration in range(num_value_iterations):
                    for s in range(env.nS):
                        for a in range(env.nA):
                            if nSA[s][a] >= m:
                                sum_val = 0.0
                                for ns in range(env.nS):
                                    sum_val += (float(model_P[s][a][ns]) * np.max(Q[ns]))
                                Q[s][a] = float(model_R[s][a]) + (float(gamma) * sum_val)

            # let next state be current state, and go back up
            state = nextstate

            # but if done = True meaning that nextstate was goal state, then we just break the episode here
            if done:
                break

        total_rewards += episode_reward
        running_average_rewards.append(float(total_rewards) / (episode + 1))
    
    ########################################################
    #                    END YOUR CODE                     #
    ########################################################
    return Q