from train_policy import PolicyModel, TrainingConfig
import torch
import numpy as np
from state import ACTION_COUNT, TaxiStateTracker


class PolicyAgent:
    def __init__(self, path):
        # Initialize the state tracker from your refactored state module
        self.state_tracker = TaxiStateTracker()
        
        # Create policy model with default configuration
        config = TrainingConfig()
        self.policy = PolicyModel(
            input_dim=self.state_tracker.state_dimension, 
            output_dim=ACTION_COUNT,
            config=config  # Use the architecture from your refactored model
        )
        
        # Load the trained model
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))
        self.policy.eval()
        
        # Track episode for auto-reset
        self._last_observation = None
    
    @torch.no_grad()
    def get_action(self, obs):
        # Auto-detect new episode by checking if observation is significantly different
        # This specifically looks for a change in taxi position which indicates a new episode
        if self._is_new_episode(obs):
            self.state_tracker.initialize()
        
        # Store current observation for next comparison - handle any data type
        self._last_observation = tuple(obs) if isinstance(obs, (list, np.ndarray)) else obs
        
        # Process the observation through the state tracker
        state, info = self.state_tracker.process_observation(obs)
        
        # Use the select_action method with deterministic=True for inference
        action, _ = self.policy.select_action(state, deterministic=True)
        
        # Update the state tracker with the selected action
        self.state_tracker.update_after_action(action)
        
        return action
    
    def _is_new_episode(self, obs):
        """Detect if this is likely the start of a new episode"""
        # First observation of the program
        if self._last_observation is None:
            return True
        
        try:
            # Check if taxi position has changed significantly from last observation
            # In normal gameplay, the taxi can only move one cell at a time
            # If it jumps to a completely different position, we likely have a new episode
            taxi_row, taxi_col = obs[0], obs[1]
            last_taxi_row, last_taxi_col = self._last_observation[0], self._last_observation[1]
            
            manhattan_distance = abs(taxi_row - last_taxi_row) + abs(taxi_col - last_taxi_col)
            
            # If manhattan distance is > 1, we've likely teleported to a new episode
            if manhattan_distance > 1:
                return True
            
            # Check if passenger status changed unexpectedly
            # If we previously had a passenger but now don't without a DROP action
            if hasattr(self.state_tracker, 'previous_action') and hasattr(self.state_tracker, 'passenger_onboard'):
                if (self.state_tracker.passenger_onboard and 
                    not self._passenger_nearby(obs) and
                    (self.state_tracker.previous_action is None or self.state_tracker.previous_action.value != 5)):  # 5 = DROP
                    return True
        except (IndexError, TypeError, AttributeError):
            # If anything goes wrong with the comparison, assume it's a new episode
            return True
        
        return False
    
    def _passenger_nearby(self, obs):
        """Check if passenger is nearby in current observation"""
        try:
            return obs[14] == 1  # Passenger nearby indicator is at index 14
        except (IndexError, TypeError):
            return False  # Default if we can't determine