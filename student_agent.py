import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from policy_agent import PolicyAgent

agent=PolicyAgent('checkpoints/policy_model.pth')

def get_action(obs):
    return agent.get_action(obs)