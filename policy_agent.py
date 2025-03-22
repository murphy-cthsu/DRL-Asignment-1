from train_policy import PolicyModel 
import torch
from state import ACTION_COUNT, TaxiStateTracker  # Your refactored state classes

class PolicyAgent:
    def __init__(self, path):
        # Initialize the state tracker from your refactored state module
        self.state_tracker = TaxiStateTracker()
        
        # Create policy model with default configuration
        self.policy = PolicyModel(
            input_dim=self.state_tracker.state_dimension, 
            output_dim=ACTION_COUNT, 
            config='configs/default_config.yaml' # Use the architecture from your refactored model
        )
        
        # Load the trained model
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))
        self.policy.eval()

    @torch.no_grad()
    def get_action(self, obs):
        # Process the observation through the state tracker
        state, info = self.state_tracker.process_observation(obs)
        
        # Use the select_action method with deterministic=True for inference
        action, _ = self.policy.select_action(state, deterministic=True)
        
        # Update the state tracker with the selected action
        self.state_tracker.update_after_action(action)
        
        return action