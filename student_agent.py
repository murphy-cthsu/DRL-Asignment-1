import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random


from policy_agent import PolicyAgent

agent=PolicyAgent('trained_models/policy_model_04.pth')

def get_action(obs):

    return agent.get_action(obs)