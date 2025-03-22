from collections import defaultdict
from enum import IntEnum
from types import SimpleNamespace
from typing import Optional, Tuple, List, Dict, Any
import torch

def distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Keep the action size constant
ACTION_COUNT = 6


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


def enhance_reward(
    original_reward: float, 
    current_info: SimpleNamespace, 
    next_info: SimpleNamespace
) -> float:
    """
    Enhance the environment reward with additional signals
    
    Args:
        original_reward: Original reward from environment
        current_info: Current state info
        next_info: Next state info
        
    Returns:
        enhanced_reward: Modified reward with additional signals
    """
    # Modify standard environment rewards
    if original_reward == 50 - 0.1:  # Successfully completed task
        enhanced_reward = 50
    elif original_reward == -10.1:  # Invalid pickup/dropoff
        enhanced_reward = -50
    elif original_reward == -5.1:  # Collision with obstacle
        enhanced_reward = -50
    elif original_reward == -0.1:  # Standard movement
        enhanced_reward = -0.1
    else:
        enhanced_reward = original_reward
    
    # Reward for successful pickup
    if not current_info.passenger_onboard and next_info.passenger_onboard:
        enhanced_reward += 30
    
    # Penalty for invalid dropoff
    elif (current_info.passenger_onboard and 
          not next_info.passenger_onboard and 
          not current_info.can_dropoff_passenger):
        enhanced_reward -= 32
    
    # Reward for discovering new station information
    unknown_count_before = sum(1 for t in current_info.location_types 
                              if t == StationState.UNEXPLORED)
    unknown_count_after = sum(1 for t in next_info.location_types 
                             if t == StationState.UNEXPLORED)
    
    if unknown_count_before > unknown_count_after:
        enhanced_reward += 16
    
    # Reward for moving toward target
    if current_info.target_location == next_info.target_location:
        distance_before = distance(current_info.taxi_position, current_info.target_location)
        distance_after = distance(next_info.taxi_position, next_info.target_location)
        distance_progress = distance_before - distance_after
        enhanced_reward += 0.1 * distance_progress
    
    return enhanced_reward