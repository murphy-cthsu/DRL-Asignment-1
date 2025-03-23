from train_policy import PolicyModel, TrainingConfig
import torch
import numpy as np
import random
from state import ACTION_COUNT, TaxiStateTracker, MovementAction


class PolicyAgent:
    def __init__(self, path):
        # Initialize the state tracker from your refactored state module
        self.state_tracker = TaxiStateTracker()
        
        # Create policy model with default configuration
        config = TrainingConfig()
        self.policy = PolicyModel(
            input_dim=self.state_tracker.state_dimension, 
            output_dim=ACTION_COUNT,
            config=config
        )
        
        # Load the trained model
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))
        self.policy.eval()
        
        # Track episode for auto-reset
        self._last_observation = None
        
        # Exploration parameters
        self.random_action_prob = 0.02  # Base probability for random actions
        self.last_random_action_step = 0
        self.random_action_cooldown = 3  # Steps to wait between random actions
    
    @torch.no_grad()
    def get_action(self, obs):
        # Auto-detect new episode by checking if observation is significantly different
        if self._is_new_episode(obs):
            self.state_tracker.initialize()
        
        # Store current observation for next comparison - handle any data type
        self._last_observation = tuple(obs) if isinstance(obs, (list, np.ndarray)) else obs
        
        # Process the observation through the state tracker
        state, info = self.state_tracker.process_observation(obs)
        
        # Decide if we should take a random action to break out of loops
        should_explore = self._should_take_random_action(info)
        
        if should_explore:
            # Take random action but avoid the problematic ones if detected
            action = self._select_smart_random_action(info)
            # Record the step we took a random action
            self.last_random_action_step = info.time_step
        else:
            # Use the policy model as normal
            action, _ = self.policy.select_action(state, deterministic=False)
        
        # Update the state tracker with the selected action
        self.state_tracker.update_after_action(action)
        
        return action
    
    def _should_take_random_action(self, info):
        """Determine if a random action should be taken to break out of loops"""
        # Base chance of random action
        random_action_prob = self.random_action_prob
        
        # Don't take random actions too frequently
        if info.time_step - self.last_random_action_step < self.random_action_cooldown:
            return False
        
        # Increase probability if we detect repetitive behavior
        if info.in_repetitive_loop:
            random_action_prob += 0.4
        
        # Increase probability if we're stuck at the same position
        if info.stuck_at_location:
            random_action_prob += 0.3
        
        # Increase probability for repeated invalid pickup attempts
        if info.invalid_pickup_count > 1:
            random_action_prob += 0.15 * info.invalid_pickup_count
        
        # Increase probability for repeated invalid dropoff attempts
        if info.invalid_dropoff_count > 1:
            random_action_prob += 0.15 * info.invalid_dropoff_count
        
        # Increase probability if we're frequently visiting the same state
        current_pos_visits = info.position_visit_frequency[(*info.taxi_position, info.passenger_onboard)]
        if current_pos_visits > 3:
            random_action_prob += 0.1 * (current_pos_visits - 3)
        
        # Cap the probability at 0.9
        random_action_prob = min(random_action_prob, 0.9)
        
        # Return True with the calculated probability
        return random.random() < random_action_prob
    
    def _select_smart_random_action(self, info):
        """Select a random action, but avoid obviously bad choices"""
        # Avoid repeatedly trying invalid pickup/dropoff
        avoid_actions = []
        
        # Avoid pickup if not at passenger location or already have passenger
        if info.invalid_pickup_count > 0 or info.passenger_onboard:
            avoid_actions.append(MovementAction.TAKE.value)
        
        # Avoid dropoff if not at destination or don't have passenger
        if info.invalid_dropoff_count > 0 or not info.passenger_onboard:
            avoid_actions.append(MovementAction.DROP.value)
        
        # Avoid moving into walls
        if info.wall_obstacles[0]:  # North wall
            avoid_actions.append(MovementAction.UP.value)
        if info.wall_obstacles[1]:  # South wall
            avoid_actions.append(MovementAction.DOWN.value)
        if info.wall_obstacles[2]:  # East wall
            avoid_actions.append(MovementAction.RIGHT.value)
        if info.wall_obstacles[3]:  # West wall
            avoid_actions.append(MovementAction.LEFT.value)
        
        # Create a list of possible actions (excluding ones to avoid)
        possible_actions = [a for a in range(ACTION_COUNT) if a not in avoid_actions]
        
        # If all actions are bad (shouldn't happen), just pick any action
        if not possible_actions:
            possible_actions = list(range(ACTION_COUNT))
        
        # Return a random action from the possible ones
        return random.choice(possible_actions)
    
    def _is_new_episode(self, obs):
        """Detect if this is likely the start of a new episode"""
        # First observation of the program
        if self._last_observation is None:
            return True
        
        try:
            # Check if taxi position has changed significantly from last observation
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
                    obs[14] == 0 and  # No passenger nearby
                    (self.state_tracker.previous_action is None or self.state_tracker.previous_action.value != 5)):  # 5 = DROP
                    return True
        except (IndexError, TypeError, AttributeError):
            # If anything goes wrong with the comparison, assume it's a new episode
            return True
        
        return False