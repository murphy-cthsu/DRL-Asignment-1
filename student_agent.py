import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from collections import defaultdict
from enum import IntEnum
from types import SimpleNamespace
from typing import Optional, Tuple, List, Dict, Any, Union
        
def distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Keep the action size constant
ACTION_COUNT = 6

class PolicyModel(nn.Module):
    """Enhanced policy network with improved architecture and utilities"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=[8]):
        """
        Initialize the enhanced policy network
        
        Args:
            input_dim: Dimension of state input
            output_dim: Dimension of action output
            hidden_sizes: List of hidden layer sizes
        """
        super(PolicyModel, self).__init__()
        
        # Build network layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            # Add dropout for regularization
            layers.append(nn.Dropout(0.0))
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Action head with softmax
        self.action_head = nn.Softmax(dim=-1)
        
        # Initialize weights properly
        # self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU networks
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            state: The input state
            
        Returns:
            Action probabilities
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            x = torch.tensor(state, dtype=torch.float32)
        else:
            x = state
        
        # Process through network layers
        logits = self.network(x)
        
        # Apply softmax to get probabilities
        return self.action_head(logits)
    
    def select_action(self, state: Union[List, np.ndarray], deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Select an action based on current policy
        
        Args:
            state: Current state
            deterministic: If True, select best action, otherwise sample
            
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
        """
        # Get action probabilities
        action_probs = self(state)
        
        # Create distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Sample or take best action
        if deterministic:
            action = torch.argmax(action_probs).item()
            # We still need the log prob for the chosen action
            log_prob = dist.log_prob(torch.tensor(action))
        else:
            # Sample from distribution
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
        
        return action, log_prob
    
    def save(self, path: str):
        """Save model to specified path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model from specified path"""
        self.load_state_dict(torch.load(path))


class MovementAction(IntEnum):
    """Enum representing possible taxi actions"""
    DOWN = 0    # SOUTH
    UP = 1      # NORTH
    RIGHT = 2   # EAST
    LEFT = 3    # WEST
    TAKE = 4    # PICKUP
    DROP = 5    # DROPOFF
    
    def as_vector(self) -> List[int]:
        """Convert action to one-hot encoding"""
        result = [0] * len(self.__class__)
        result[self.value] = 1
        return result
    
    @classmethod
    def zero_vector(cls) -> List[int]:
        """Generate a zero vector with appropriate length"""
        return [0] * len(cls)


class Orientation(IntEnum):
    """Enum representing relative orientations"""
    CENTER = 0
    WEST_OF = 1
    EAST_OF = 2
    NORTH_OF = 3
    SOUTH_OF = 4
    NORTHEAST = 5
    NORTHWEST = 6
    SOUTHWEST = 7
    SOUTHEAST = 8
    
    @staticmethod
    def calculate(origin: Tuple[int, int], target: Tuple[int, int]) -> int:
        """Determine orientation of target relative to origin"""
        def compare(a: int, b: int) -> int:
            if a == b:
                return 0
            return 1 if a < b else -1
            
        # Create mapping of relative position to orientation
        position_map = {
            (0, 0): Orientation.CENTER,
            (1, 0): Orientation.EAST_OF,
            (0, 1): Orientation.NORTH_OF,
            (-1, 0): Orientation.WEST_OF,
            (0, -1): Orientation.SOUTH_OF,
            (1, 1): Orientation.NORTHEAST,
            (-1, 1): Orientation.NORTHWEST,
            (-1, -1): Orientation.SOUTHWEST,
            (1, -1): Orientation.SOUTHEAST
        }
        
        dx = compare(origin[0], target[0])
        dy = compare(origin[1], target[1])
        return position_map[(dx, dy)]
    
    def as_vector(self) -> List[int]:
        """Convert orientation to one-hot encoding"""
        result = [0] * len(self.__class__)
        result[self.value] = 1
        return result


class StationState(IntEnum):
    """Enum representing types of stations"""
    UNEXPLORED = 0
    EMPTY = 1
    PASSENGER_LOCATION = 2
    DESTINATION_LOCATION = 3
    
    def as_vector(self) -> torch.Tensor:
        """Convert location type to one-hot tensor"""
        result = torch.zeros(len(self.__class__))
        result[self.value] = 1
        return result


class TaxiStateTracker:
    """Manages and tracks the state of the taxi environment"""
    
    def __init__(self):
        """Initialize the state tracker"""
        self.initialize()
    
    def initialize(self):
        """Reset all state variables to initial values"""
        self.passenger_onboard = False
        self.location_types = [StationState.UNEXPLORED] * 4
        self.passenger_location = None
        self.previous_action: Optional[MovementAction] = None
        self.position_visit_frequency = defaultdict(int)
        self.time_step = 0
        
        # Action tracking for repetitive behavior detection
        self.action_history = []
        self.invalid_pickup_count = 0
        self.invalid_dropoff_count = 0
        self.repeated_action_count = 0
        self.same_position_count = 0
        self.previous_position = None
        
        # Flags for problematic behaviors
        self.in_repetitive_loop = False
        self.stuck_at_location = False
    
    def process_observation(self, observation) -> Tuple[Tuple, SimpleNamespace]:
        """
        Process an observation from the environment
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            state: Simplified state representation as a tuple
            info: Detailed state information in a SimpleNamespace
        """
        # Extract observation components
        self._extract_observation_data(observation)
        
        # Check if agent is stuck at the same position
        if self.previous_position == self.taxi_position:
            self.same_position_count += 1
            if self.same_position_count > 3:
                self.stuck_at_location = True
        else:
            self.same_position_count = 0
            self.stuck_at_location = False
        
        # Update previous position
        self.previous_position = self.taxi_position
        
        # Update station information based on current observation
        self._update_station_knowledge()
        
        # Infer unknown station types where possible
        self._infer_unknown_stations()
        
        # Determine action possibilities
        self._determine_action_possibilities()
        
        # Update passenger location if carrying passenger
        if self.passenger_onboard:
            self.passenger_location = self.taxi_position
        
        # Determine target station
        self._determine_target_location()
        
        # Calculate directional info to target
        target_row_diff = int(self.target_location[0] > self.taxi_position[0])
        target_row_diff -= int(self.target_location[0] < self.taxi_position[0])
        target_col_diff = int(self.target_location[1] > self.taxi_position[1])
        target_col_diff -= int(self.target_location[1] < self.taxi_position[1])
        
        # Track state visit frequency
        self.position_visit_frequency[(*self.taxi_position, self.passenger_onboard)] += 1
        
        # Check for repetitive patterns in recent actions (last 6 actions)
        if len(self.action_history) >= 6:
            # Check for repeating patterns (like A-B-A-B-A-B or A-A-A-A-A-A)
            if (self._is_alternating_pattern() or self._is_same_action_repeated()):
                self.in_repetitive_loop = True
            else:
                self.in_repetitive_loop = False
        
        # Increment time step
        self.time_step += 1
        
        # Create compact state representation
        state = (
            *self.wall_obstacles,
            self.can_pickup_passenger,
            self.can_dropoff_passenger,
            self.target_location[0] - self.taxi_position[0],
            self.target_location[1] - self.taxi_position[1],
        )
        
        # Create comprehensive state info
        info = SimpleNamespace(**vars(self))
        
        return state, info
    
    def _is_alternating_pattern(self):
        """Check for alternating action patterns like A-B-A-B"""
        if len(self.action_history) < 4:
            return False
        
        # Check last 4 actions for A-B-A-B pattern
        recent = self.action_history[-4:]
        return (recent[0] == recent[2] and 
                recent[1] == recent[3] and 
                recent[0] != recent[1])
    
    def _is_same_action_repeated(self):
        """Check if the same action is repeated many times"""
        if len(self.action_history) < 3:
            return False
        
        # Check if last 3 actions are the same
        recent = self.action_history[-3:]
        return recent.count(recent[0]) == 3
    
    def update_after_action(self, action: int):
        """Update state after an action is taken"""
        # Store previous action
        self.previous_action = MovementAction(action)
        self.action_history.append(action)
        
        # Keep history bounded
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Check for repeated actions of the same type
        if len(self.action_history) >= 2 and self.action_history[-1] == self.action_history[-2]:
            self.repeated_action_count += 1
        else:
            self.repeated_action_count = 0
        
        # Track invalid pickup/dropoff attempts
        if action == MovementAction.TAKE and not self.can_pickup_passenger:
            self.invalid_pickup_count += 1
        else:
            # Reset counter if we're not trying invalid pickup
            if action != MovementAction.TAKE:
                self.invalid_pickup_count = 0
        
        if action == MovementAction.DROP and not self.can_dropoff_passenger:
            self.invalid_dropoff_count += 1
        else:
            # Reset counter if we're not trying invalid dropoff
            if action != MovementAction.DROP:
                self.invalid_dropoff_count = 0
        
        # Update passenger status based on action
        if action == MovementAction.TAKE and self.taxi_position == self.passenger_location:
            self.passenger_onboard = True
            # Reset invalid pickup counter after successful pickup
            self.invalid_pickup_count = 0
        elif action == MovementAction.DROP:
            self.passenger_onboard = False
    
    def _extract_observation_data(self, observation):
        """Extract and store observation components"""
        self.taxi_position = (observation[0], observation[1])
        self.station_positions = [
            (observation[2], observation[3]),   # R station
            (observation[4], observation[5]),   # G station
            (observation[6], observation[7]),   # Y station
            (observation[8], observation[9]),   # B station
        ]
        self.wall_obstacles = (
            observation[10],  # North wall
            observation[11],  # South wall
            observation[12],  # East wall
            observation[13],  # West wall
        )
        self.passenger_nearby = observation[14]
        self.destination_nearby = observation[15]
    
    def _update_station_knowledge(self):
        """Update knowledge about stations based on current position and observations"""
        # If we're at a station, update its type
        if self.taxi_position in self.station_positions:
            station_index = self.station_positions.index(self.taxi_position)
            
            # Check if this station is the destination
            if self.destination_nearby:
                # Only update if we haven't found a destination yet or if this was previously marked as destination
                if (StationState.DESTINATION_LOCATION not in self.location_types or 
                    self.location_types[station_index] == StationState.DESTINATION_LOCATION):
                    # Clear any previous destination marking (if any other station was incorrectly marked)
                    for i, loc_type in enumerate(self.location_types):
                        if i != station_index and loc_type == StationState.DESTINATION_LOCATION:
                            # Downgrade to empty if it was falsely marked as destination
                            self.location_types[i] = StationState.EMPTY
                    
                    # Mark current station as destination
                    self.location_types[station_index] = StationState.DESTINATION_LOCATION
                
            # If passenger is nearby (and we don't know passenger location yet)
            elif (self.passenger_nearby and 
                (self.passenger_location is None or self.taxi_position == self.passenger_location)):
                self.passenger_location = self.taxi_position
                self.location_types[station_index] = StationState.PASSENGER_LOCATION
                
            # Otherwise if still unexplored, mark as empty
            elif self.location_types[station_index] == StationState.UNEXPLORED:
                self.location_types[station_index] = StationState.EMPTY
    
    def _infer_unknown_stations(self):
        """Use logic to infer types of unknown stations"""
        # Count known station types
        known_count = sum(1 for loc_type in self.location_types 
                          if loc_type != StationState.UNEXPLORED)
        
        # Case 1: If we know 3 stations, deduce the 4th
        if known_count == 3:
            # Find index of unknown station
            unknown_index = self.location_types.index(StationState.UNEXPLORED)
            # print(self.location_types)
            # Calculate what the unknown station must be (sum of all types = 7)
            deduced_type = StationState(7 - sum(self.location_types))
            self.location_types[unknown_index] = deduced_type
            
            # If we deduced passenger location, record it
            if deduced_type == StationState.PASSENGER_LOCATION:
                self.passenger_location = self.station_positions[unknown_index]
        
        # Case 2: If we know passenger and destination, remaining are empty
        elif (known_count == 2 and 
              StationState.PASSENGER_LOCATION in self.location_types and
              StationState.DESTINATION_LOCATION in self.location_types):
            
            # Mark all unexplored stations as empty
            for i, loc_type in enumerate(self.location_types):
                if loc_type == StationState.UNEXPLORED:
                    self.location_types[i] = StationState.EMPTY
    
    def _determine_action_possibilities(self):
        """Determine if pickup/dropoff actions are possible"""
        # Can pickup if at passenger location and not carrying
        self.can_pickup_passenger = int(
            not self.passenger_onboard and 
            self.taxi_position == self.passenger_location
        )
        
        # Can dropoff if carrying passenger and at destination
        self.can_dropoff_passenger = int(
            self.passenger_onboard and
            self.destination_nearby and
            self.taxi_position in self.station_positions
        )
    
    def _determine_target_location(self):
        """Determine the current target location based on state"""
        if not self.passenger_onboard:
            # If we know passenger location, target it
            if self.passenger_location is not None:
                self.target_location = self.passenger_location
            else:
                # Otherwise, find nearest unexplored station
                unexplored_stations = [
                    station for station, loc_type in zip(self.station_positions, self.location_types)
                    if loc_type == StationState.UNEXPLORED
                ]
                
                # Select nearest unexplored station
                self.target_location = min(
                    unexplored_stations,
                    key=lambda station: distance(self.taxi_position, station),
                    default=self.station_positions[0]  # Default to first station if none unexplored
                )
        else:
            # If carrying passenger, head to destination if known
            if StationState.DESTINATION_LOCATION in self.location_types:
                destination_index = self.location_types.index(StationState.DESTINATION_LOCATION)
                self.target_location = self.station_positions[destination_index]
            else:
                # Otherwise, explore nearest unexplored station
                unexplored_stations = [
                    station for station, loc_type in zip(self.station_positions, self.location_types)
                    if loc_type == StationState.UNEXPLORED
                ]
                
                # Select nearest unexplored station
                self.target_location = min(
                    unexplored_stations,
                    key=lambda station: distance(self.taxi_position, station),
                    default=self.station_positions[0]  # Default to first station if none unexplored
                )
    
    def update_after_action(self, action: int):
        """Update state after an action is taken"""
        self.previous_action = MovementAction(action)
        
        # Update passenger status based on action
        if action == MovementAction.TAKE and self.taxi_position == self.passenger_location:
            self.passenger_onboard = True
        elif action == MovementAction.DROP:
            self.passenger_onboard = False
    
    @property
    def state_dimension(self) -> int:
        """Return the dimensionality of the state representation"""
        return len(TaxiStateTracker().process_observation([0] * 16)[0])


class PolicyAgent:
    def __init__(self, path):
        # Initialize the state tracker from your refactored state module
        self.state_tracker = TaxiStateTracker()
        

        self.policy = PolicyModel(
            input_dim=self.state_tracker.state_dimension, 
            output_dim=ACTION_COUNT,
            hidden_sizes=[8]
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
        # should_explore = False
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

agent=PolicyAgent('trained_models/policy_model_04.pth')

def get_action(obs):

    return agent.get_action(obs)