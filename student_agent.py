import numpy as np
import pickle
import random
import gym
from collections import defaultdict
from Q_agent import TaxiAgent

agent= TaxiAgent(epsilon=0.0)
agent.load_policy('taxi_agent_qvalues.npy')
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
    # This function assumes a global 'agent' variable has been created
    # and trained before being called by the environment runner
    global agent
    return agent.get_action(state)