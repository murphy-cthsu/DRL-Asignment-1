import numpy as np
import pickle
import random
import gym
from collections import defaultdict
from Q_agent import TaxiAgent


# print(agent.Q)
def get_action(state):
    """
    Function to be used by the environment runner.
    This function should be in a module named 'student_agent.py'
    
    Parameters:
    -----------
    state : tuple
        The current state
        
    Returns:
    --------
    int
        The selected action
    """
    agent= TaxiAgent(epsilon=0.0)
    agent.load_policy('taxi_agent_qvalues.npy')
    return agent.get_action(state)