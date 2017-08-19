# Author: Jay H Patel (jayhp9@stanford.edu)
# Date created: 08/18/2017
# Python version: 3

import numpy as np
import random

from mdp import Env, Agent, State
from student import Student, AdvancedStudentConfig, OKStudentConfig, BeginnerStudentConfig, OKToAdvStudentConfig
from utils import print_Q, modified_greedy_action

"""
You will be implementing the following methods in this file:
  StandardRun.do (4-8 lines)
  IntelligentRun.do (5-8 lines)
  IntelligentRun.do_calibration_questions (11-18 lines)
  IntelligentRun.do_post_calibration_questions (15-23 lines)
"""

class Runs:
    def __init__(self, student_config, run_config, num_runs=10):
        self.runs = []
        self.run_config = run_config
        self.student_config = student_config
        for r in range(num_runs):
            if type(run_config) is IntelligentRunConfig:
                self.runs.append(IntelligentRun(student_config.__class__, run_config))
            elif type(run_config) is StandardRunConfig:
                self.runs.append(StandardRun(student_config.__class__, run_config))

    def do(self):
        print("RUNNING {} {} students with run config: {}".format(len(self.runs), str(self.student_config), str(self.run_config)))
        for run in self.runs:
            run.do()

        if type(self.run_config) is IntelligentRunConfig:
            self.print_intelligent_run_stats()
        elif type(self.run_config) is StandardRunConfig:
            self.print_standard_run_stats()

    def print_intelligent_run_stats(self):
        all_calibration_rewards = []
        all_post_calibration_rewards = []
        for run in self.runs:
            all_calibration_rewards.append(np.sum(run.calibration_rewards))
            all_post_calibration_rewards.append(np.sum(run.post_calibration_rewards))
        
        print("=> {} points on average for calibration questions across {} {} students".format(np.average(all_calibration_rewards), len(self.runs), str(self.student_config)))
        print("=> {} points on average for post calibration questions across {} {} students".format(np.average(all_post_calibration_rewards), len(self.runs), str(self.student_config)))
        print("")

    def print_standard_run_stats(self):
        rewards = []
        for run in self.runs:
            rewards.append(np.sum(run.rewards))
            
        print("=> {} points on average for questions across 10 {} students".format(np.average(rewards), str(self.student_config)))
        print("")

class StandardRunConfig:
    def __init__(self, num_questions=15, difficulty=None):
        self.initial_state = State.VERY_LOW_UNDERSTANDING_STATE
        self.num_questions = num_questions
        self.nS = 5
        self.nA = 5
        self.print_stats = True
        self.difficulty = difficulty

    def __str__(self):
        difficulty = "random" if self.difficulty is None else self.difficulty
        return "{} qns of difficulty {} from initial state {}".format(self.num_questions, difficulty, self.initial_state)

class StandardRun:
    def __init__(self, student_config_class, run_config):
        self.student = Student(student_config_class())
        self.run_config = run_config
        self.state = self.run_config.initial_state

        self.rewards = []

    def do(self):
        """Presents problems of random difficulties to the student and collects the rewards in self.rewards. 
        The number of problems is given by self.run_config.num_questions. If self.run_config.difficulty 
        is defined, present all problems of that difficulty. self.student has a method called 
        'simulate_problem' that can be used to simulate the student's response to the problem.
        """
        # ===== YOUR CODE STARTS HERE (4-8 LINES) =====
        for qn in range(self.run_config.num_questions):
            diff = random.randint(0, 4) if self.run_config.difficulty is None else self.run_config.difficulty
            r = self.student.simulate_problem(diff)
            self.rewards.append(r)
        # ===== YOUR CODE ENDS HERE =====

class IntelligentRunConfig:
    def __init__(self, num_calibration=5, num_post_calibration=10, learning_algo="qlearning", sarsa_update=True):
        self.initial_state = State.VERY_LOW_UNDERSTANDING_STATE
        self.nS = 5
        self.nA = 5
        self.episode_length = 10
        self.num_calibration = num_calibration
        self.num_post_calibration = num_post_calibration
        
        self.learning_algo = learning_algo
        self.sarsa_update = sarsa_update
        self.post_calibration_eps = 0.2
        self.print_stats = True

    def __str__(self):
        s = "{} calibration qns and {} post-calibration qns from initial state {}".format(self.num_calibration, self.num_post_calibration, self.initial_state)
        if self.num_post_calibration > 0:
            s +=  " with SARSA update {} and {} for initial agents".format("ON" if self.sarsa_update else "OFF", self.learning_algo)
        return s

class IntelligentRun:
    def __init__(self, student_config_class, run_config):
        self.student = Student(student_config_class())
        self.run_config = run_config
        self.state = self.run_config.initial_state
        
        self.agent_advanced = self.setup_agent(AdvancedStudentConfig())
        self.agent_ok = self.setup_agent(OKStudentConfig())
        self.agent_beginner = self.setup_agent(BeginnerStudentConfig())
        
        # initiate priors: advanced (idx 0), ok (idx 1), beginner (idx 2)
        self.probabilities = [float(1/3), float(1/3), float(1/3)]

        self.calibration_rewards = []
        self.calibration_problems = []
        self.post_calibration_rewards = []
        self.post_calibration_problems = []

    def setup_agent(self, student_config):
        student = Student(student_config)
        env = Env(student, self.run_config.initial_state, self.run_config.nS, self.run_config.nA, self.run_config.episode_length)
        agent = Agent(env, algo=self.run_config.learning_algo)
        agent.learn_q()
        agent.compute_probs_reward_per_state()
        return agent

    def do(self):
        """If self.run_config.num_calibration is greater than zero, calls self.do_calibration_questions.
        Likewise, if self.run_config.num_post_calibration is greater than zero, calls
        self.do_post_calibration_questions.

        First, we will run the calibration questions to obtain a probability distribution over the three
        student types, reflecting the model's understanding of how likely the student is to be one of
        the types. For example, the probability of advanced student might be 0.8, the probability of ok student
        might be 0.15 and the probability of beginner student might be 0.05. You will be implementing this
        logic in do_calibration_questions, using an algorithm called Thompson Sampling.

        After finishing the calibration questions, we take a weighted average of the Q values for each student
        type according to the learned probability distribution explained above. This will give us a new array of
        Q values that represents the current student. Using this new combined array, we will then run the post
        calibration questions.
        """
        # ===== YOUR CODE STARTS HERE (5-8 LINES) =====
        # run calibration questions and collect belief distribution after that
        if self.run_config.num_calibration > 0:
            self.do_calibration_questions()

        if self.run_config.num_post_calibration > 0:
            # compute combined Q values for new student
            Q_combined = float(self.probabilities[0])*self.agent_advanced.Q + float(self.probabilities[1])*self.agent_ok.Q + float(self.probabilities[2])*self.agent_beginner.Q

            self.do_post_calibration_questions(Q_combined, self.run_config.sarsa_update)
        # ===== YOUR CODE ENDS HERE =====
        
        if self.run_config.print_stats:
            all_rewards = self.calibration_rewards + self.post_calibration_rewards
            all_rewards.insert(len(self.calibration_rewards), " - ") # mark end of calibration
            all_problem_difficulties = self.calibration_problems + self.post_calibration_problems
            all_problem_difficulties.insert(len(self.calibration_problems), " - ") # mark end of calibration
            if type(self.student.config) is OKToAdvStudentConfig:
                all_rewards.insert(self.student.config.improvement_threshold, " ^ ") # mark change of student level
                all_problem_difficulties.insert(self.student.config.improvement_threshold, " ^ ") # mark change of student level
            print("rewards:", str(all_rewards))
            print("problem difficulties:", str(all_problem_difficulties))
            print("")

    def do_calibration_questions(self):
        """Present calibration problems to the student and perform Thompson Sampling updates as we go, to learn
        the mostly type of the student (Beginner, OK, or Advanced).

        Initially, we don't know what type the student is, so the probabilities are split 1/3, 1/3, 1/3.
        We sample a student type according to these probabilities, and present a problem and record the reward
        and next state. Using this reward, we perform a Bayesian update to the probabilities, according to 
        how likely each student type is to have given rise to that reward in the current state.

        We do this for each of the calibration problems and end up with a probability distribution over all
        student types.
        """
        # ===== YOUR CODE STARTS HERE (11-18 LINES) =====
        for qn in range(self.run_config.num_calibration):
            # sample a student
            sampled_agent = np.random.choice([self.agent_advanced, self.agent_ok, self.agent_beginner], p=self.probabilities)

            # choose action from the sampled Q, and observe reward at that state
            action = sampled_agent.greedy_action(self.state)
            r = self.student.simulate_problem(action)
            self.calibration_rewards.append(r)
            self.calibration_problems.append(action)

            # first compute normalization factor of Thompson Sampling
            Z = (self.agent_advanced.reward_probabilities[action][r] * float(self.probabilities[0])) + (self.agent_ok.reward_probabilities[action][r] * float(self.probabilities[1])) + (self.agent_beginner.reward_probabilities[action][r] * float(self.probabilities[2]))

            # perform bayesian update for advanced student
            self.probabilities[0] = (self.agent_advanced.reward_probabilities[action][r] * float(self.probabilities[0])) / float(Z)

            # perform bayesian update for ok student
            self.probabilities[1] = (self.agent_ok.reward_probabilities[action][r] * float(self.probabilities[1])) / float(Z)

            # perform bayesian update for beginner student
            self.probabilities[2] = (self.agent_beginner.reward_probabilities[action][r] * float(self.probabilities[2])) / float(Z)

            self.state = action
        # ===== YOUR CODE ENDS HERE =====

    def do_post_calibration_questions(self, Q, update):
        """Present post calibration questions according to the combined Q values passed in. Again, for each
        problem, record the next state, reward.

        When choosing the action, we use the modified_greedy_action util function to periodically present
        a problem from a higher state (i.e. higher difficulty). This allows our model to probe the student's
        understanding to see if it has since risen.

        If update is true, we update the Q values based on the reward obtained from presenting a problem.
        The formula is given by:
            Q[state][action] = Q[state][action] + 0.6*(reward + 0.99*Q[next_state][next_action] - Q[state][action])
        This is useful because it keeps our Q values adaptable and representative of the current competence
        of the student.

        Parameters
        ----------
        Q: np.array
            An array of shape [env.nS x env.nA] representing state, action values
        update: boolean
            Whether or not to continually update the Q values
        """
        # ===== YOUR CODE STARTS HERE (15-23 LINES) =====
        prev_state = None
        prev_action = None
        prev_reward = None

        for qn in range(self.run_config.num_post_calibration):
            action = modified_greedy_action(self.run_config.post_calibration_eps, Q, self.state)
            if update and prev_state is not None and prev_action is not None and prev_reward is not None:
                old = Q[prev_state][prev_action]
                Q[prev_state][prev_action] = Q[prev_state][prev_action] + 0.6*(prev_reward + 0.99*Q[self.state][action] - Q[prev_state][prev_action])
            r = self.student.simulate_problem(action)
            self.post_calibration_rewards.append(r)
            self.post_calibration_problems.append(action)

            # record for SARSA update
            prev_state = self.state
            prev_action = action
            prev_reward = r

            # move to next state
            self.state = action
        # ===== YOUR CODE ENDS HERE =====
