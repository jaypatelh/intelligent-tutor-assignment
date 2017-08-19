# Author: Jay H Patel (jayhp9@stanford.edu)
# Date created: 08/18/2017
# Python version: 3

import os
import pytz
import random
from datetime import timedelta
import numpy as np

from operator import itemgetter

from run import Runs, IntelligentRunConfig, StandardRunConfig
from student import OKToAdvStudentConfig, AdvancedStudentConfig, OKStudentConfig, BeginnerStudentConfig

"""
Here we define and execute our tutoring simulations (i.e. a Runs object, as defined in run.py),
and output the results.

You will be implementing the following methods in this file:
  random_problems (4-8 lines)
  intelligently_chosen_problems (12-16 lines)
"""

def random_problems():
    """Run simulations where problems are randomly shown to student and collect statistics on
    how often they are answered correctly (based on the reward observed).

    First create a StandardRunConfig object with 10 questions. Then, create a simulation (i.e. Runs)
    for each type of student (BeginnerStudentConfig, OKStudentConfig, and AdvancedStudentConfig), 
    pass in the StandardRunConfig object. Finally, execute the simulation by calling the do() 
    method.

    The do() method will simulate 10 students of the specified type, and 10 random problems per student,
    and will print the average performance of the students on those problems.

    You should expect to see between 2 and 4 points on the 10 problems. Each problem is worth 1 point, so the
    best case is 10 points. However, since this test presents problems randomly, we don't always show the
    most appropriately challenging problem, so more often, the reward obtained is 0.
    """
    # ===== YOUR CODE STARTS HERE (4-8 LINES) =====
    standard = StandardRunConfig(num_questions=10)
    Runs(AdvancedStudentConfig(), standard).do()
    Runs(OKStudentConfig(), standard).do()
    Runs(BeginnerStudentConfig(), standard).do()
    # ===== YOUR CODE ENDS HERE =====

def intelligently_chosen_problems():
    """Run simulations where problems are intelligently chosen and collect statistics on
    how often they are answered correctly (based on the rewards observed).

    Implement three different variations:
    1. Only 5 calibration problems (i.e. where we are actively updating our student beliefs using Thompson Sampling)
    2. 5 calibration problems, followed by 10 non-calibration problems (no longer updating student beliefs)
    3. 5 calibration problems, followed by 30 non-calibration problems where we also update Q values based on
        newly observed rewards (set 'sarsa_update' argument to True)

    We can specify all these variations by creating different versions of the IntelligentRunConfig
    object.

    For Variations 1 and 2, create a simulation (i.e. Runs object) for each type of student (BeginnerStudentConfig, 
    OKStudentConfig, and AdvancedStudentConfig), pass in the IntelligentRunConfig. Finally, execute the 
    simulation by calling the do() method.

    For Variation 3, we are interested in simulating how well our model adapts when a student improves midway i.e.
    after answering some questions, the student goes from being OK to Advanced. Ideally, we would like our model
    to detect this and adapt by showing increasingly harder questions. We will use the OKToAdvStudentConfig()
    student config here (which you will also implement in student.py), and set the 'sarsa_update' argument
    to IntelligentRunConfig to True.

    Expected results
    ----------------
    For Variation 1, you should expect 3-4 points on the 5 calibration problems. This is better than a random
    algorithm. Since there are no post-calibration problems here, the points printed are 0. Each problem
    is worth 1 point.

    For Variation 2, you should expect 3-4 points on the 5 calibration problems, as above. Then, you should expect
    between 8-10 points on the following 10 post-calibration problems. This is great performance and reflects that
    our model has correctly identified the student's level of understanding.

    For Variation 3, you should expect 3-4 points on the 5 calibration problems, as above. Then, you should
    expect around 20+ points in the case where we are performing SARSA updates continually, and around 18+
    points in the case where we don't perform the SARSA updates in the following 30 post calibration problems. 
    You may not get these exact numbers, but the points in the former case should be higher than 
    the latter.
    """
    # ===== YOUR CODE STARTS HERE (12-16 LINES) =====
    # Variation 1
    only_calibration = IntelligentRunConfig(num_post_calibration=0)
    Runs(AdvancedStudentConfig(), only_calibration).do()
    Runs(OKStudentConfig(), only_calibration).do()
    Runs(BeginnerStudentConfig(), only_calibration).do()

    # Variation 2
    steady_after_calibration = IntelligentRunConfig(sarsa_update=False)
    Runs(AdvancedStudentConfig(), steady_after_calibration).do()
    Runs(OKStudentConfig(), steady_after_calibration).do()
    Runs(BeginnerStudentConfig(), steady_after_calibration).do()

    # Variation 3
    sarsa_update_improving_student = IntelligentRunConfig(sarsa_update=True, num_post_calibration=30)
    Runs(OKToAdvStudentConfig(), sarsa_update_improving_student).do()

    no_sarsa_update_improving_student = IntelligentRunConfig(sarsa_update=False, num_post_calibration=30)
    Runs(OKToAdvStudentConfig(), no_sarsa_update_improving_student).do()
    # ===== YOUR CODE ENDS HERE =====

if __name__ == '__main__':
    #random_problems()
    intelligently_chosen_problems()
