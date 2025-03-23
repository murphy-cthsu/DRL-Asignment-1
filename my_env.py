from collections import defaultdict
from typing import Literal, Tuple, Dict, List, Set, Optional, Any
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random


class TrainingTaxiEnv:
    """
    A custom Taxi environment for training agents with various grid sizes and difficulty levels.
    The agent must navigate to pick up a passenger and drop them off at a destination.
    """
    
    # Action space constants
    MOVE_SOUTH = 0
    MOVE_NORTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    PICKUP = 4
    DROPOFF = 5
    
    # Action descriptions for rendering
    ACTION_NAMES = [
        "Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"
    ]
    
    # Station labels
    STATION_LABELS = ['R', 'G', 'Y', 'B']
    
    def __init__(
        self,
        n: int = 5,
        max_fuel: int = 50,
        obstacle_prob: float = 0.2,
        difficulty: Literal['easy', 'normal', 'hard'] = 'normal'
    ):
        """
        Initialize the taxi environment.
        
        Args:
            n: Grid size (n x n)
            max_fuel: Maximum fuel the taxi can have
            obstacle_prob: Probability of obstacles (for custom generation)
            difficulty: Difficulty level affecting number of obstacles
        """
        self.grid_size = n
        self.fuel_limit = max_fuel
        self.difficulty = difficulty
        self.obstacle_prob = obstacle_prob
        
        # Define action and observation spaces
        self.action_space = np.arange(6)
        
        # Setup environment
        self.stations = []
        self.taxi_pos = (0, 0)
        self.passenger_loc = (0, 0)
        self.destination = (0, 0)
        self.passenger_picked_up = False
        self.current_fuel = 0
        self.obstacles = set()
    
    def reset(self, random_stations: bool = True) -> Tuple[tuple, Dict]:
        """
        Reset the environment to a new initial state.
        
        Args:
            random_stations: Whether to place stations randomly
            
        Returns:
            Tuple of (observation, info)
        """
        # Set the number of obstacles based on difficulty
        obstacles_by_difficulty = {
            'easy': defaultdict(lambda: 0),
            'normal': {
                10: 10, 9: 8, 8: 6, 7: 4, 6: 3, 5: 3
            },
            'hard': {
                10: 20, 9: 12, 8: 8, 7: 6, 6: 5, 5: 5
            }
        }
        self.n_obstacle = obstacles_by_difficulty[self.difficulty].get(
            self.grid_size, int(self.obstacle_prob * self.grid_size * self.grid_size)
        )
        
        # Continue generating environments until we get a valid one
        while True:
            # Initialize all positions as available
            available_positions = [
                (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            ]
            
            # Place stations
            self.stations = []
            if random_stations:
                # Randomly place stations with buffer zones
                for _ in range(4):
                    if not available_positions:
                        break
                    
                    x, y = random.choice(available_positions)
                    self.stations.append((x, y))
                    
                    # Remove station and surrounding positions from available positions
                    for dx, dy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
                        if (x + dx, y + dy) in available_positions:
                            available_positions.remove((x + dx, y + dy))
            else:
                # Place stations at corners
                self.stations = [
                    (0, 0), 
                    (0, self.grid_size - 1), 
                    (self.grid_size - 1, 0), 
                    (self.grid_size - 1, self.grid_size - 1)
                ]
                
                # Remove stations from available positions
                for station in self.stations:
                    if station in available_positions:
                        available_positions.remove(station)
            
            # Reset fuel and passenger state
            self.current_fuel = self.fuel_limit
            self.passenger_picked_up = False
            
            # Update available positions
            available_positions = [
                (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                if (x, y) not in self.stations
            ]
            
            # Place taxi
            self.taxi_pos = random.choice(available_positions)
            
            # Place passenger at a station
            self.passenger_loc = random.choice(self.stations)
            
            # Place destination at a different station
            destination_options = [s for s in self.stations if s != self.passenger_loc]
            self.destination = random.choice(destination_options)
            
            # Update available positions again
            available_positions = [
                (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                if (x, y) not in self.stations and (x, y) != self.taxi_pos
            ]
            
            # Place obstacles
            obstacle_count = min(len(available_positions), self.n_obstacle)
            self.obstacles = set(random.sample(available_positions, obstacle_count))
            
            # Check if the environment is valid (taxi can reach passenger, and passenger can reach destination)
            if self._is_valid_environment():
                break
        
        return self.get_observation(), {}
    
    def _is_valid_environment(self) -> bool:
        """Check if the environment is valid (paths exist between key locations)"""
        return (self._is_path_exists(self.taxi_pos, self.passenger_loc) and 
                self._is_path_exists(self.passenger_loc, self.destination))
    
    def _is_path_exists(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        Check if a path exists between two points using BFS
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            True if a path exists, False otherwise
        """
        if start == end:
            return True
            
        # Define movement directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # E, S, W, N
        
        # BFS implementation
        queue = [start]
        visited = set()
        
        while queue:
            position = queue.pop(0)
            
            if position == end:
                return True
                
            if position in visited:
                continue
                
            visited.add(position)
            
            # Try each direction
            for dx, dy in directions:
                new_row, new_col = position[0] + dx, position[1] + dy
                new_position = (new_row, new_col)
                
                # Check if position is valid
                if (0 <= new_row < self.grid_size and 
                    0 <= new_col < self.grid_size and 
                    new_position not in self.obstacles):
                    queue.append(new_position)
        
        return False
    
    def step(self, action: int) -> Tuple[tuple, float, bool, bool, Dict]:
        """
        Take a step in the environment with the given action
        
        Args:
            action: Action to take (0-5)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Extract current position
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        
        # Process movement actions
        if action == self.MOVE_SOUTH:  # Move South
            next_row += 1
        elif action == self.MOVE_NORTH:  # Move North
            next_row -= 1
        elif action == self.MOVE_EAST:  # Move East
            next_col += 1
        elif action == self.MOVE_WEST:  # Move West
            next_col -= 1
            
        # Handle movement actions
        if action in [self.MOVE_SOUTH, self.MOVE_NORTH, self.MOVE_EAST, self.MOVE_WEST]:
            # Check if movement is valid
            if ((next_row, next_col) in self.obstacles or 
                not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size)):
                # Hit obstacle or boundary
                reward -= 5
            else:
                # Valid movement
                self.taxi_pos = (next_row, next_col)
                # If passenger is in taxi, update passenger location
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            # Handle pickup/dropoff actions
            if action == self.PICKUP:  # Pickup action
                if self.taxi_pos == self.passenger_loc:
                    # Valid pickup
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                else:
                    # Invalid pickup
                    reward = -10
            elif action == self.DROPOFF:  # Dropoff action
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        # Successful dropoff
                        reward += 50
                        # Task completed successfully
                        return self.get_observation(), reward - 0.1, True, False, {}
                    else:
                        # Invalid dropoff location
                        reward -= 10
                    # Release passenger
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    # No passenger to drop off
                    reward -= 10
        
        # Small time penalty for each step
        reward -= 0.1
        
        # Consume fuel
        self.current_fuel -= 1
        
        # Check if out of fuel
        if self.current_fuel <= 0:
            return self.get_observation(), reward, True, False, {}
        
        return self.get_observation(), reward, False, False, {}
    
    def get_observation(self) -> tuple:
        """
        Get the current observation of the environment
        
        Returns:
            Tuple representing the state
        """
        # Extract positions
        taxi_row, taxi_col = self.taxi_pos
        
        # Check for obstacles in each direction
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size-1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size-1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col-1) in self.obstacles)
        
        # Check if passenger is visible (in adjacent cells or current cell)
        passenger_loc_north = int((taxi_row-1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row+1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int((taxi_row, taxi_col+1) == self.passenger_loc)
        passenger_loc_west = int((taxi_row, taxi_col-1) == self.passenger_loc)
        passenger_loc_current = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_visible = (passenger_loc_north or passenger_loc_south or 
                           passenger_loc_east or passenger_loc_west or passenger_loc_current)
        
        # Check if destination is visible (in adjacent cells or current cell)
        destination_loc_north = int((taxi_row-1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row+1, taxi_col) == self.destination)
        destination_loc_east = int((taxi_row, taxi_col+1) == self.destination)
        destination_loc_west = int((taxi_row, taxi_col-1) == self.destination)
        destination_loc_current = int((taxi_row, taxi_col) == self.destination)
        destination_visible = (destination_loc_north or destination_loc_south or 
                             destination_loc_east or destination_loc_west or destination_loc_current)
        
        # Construct state tuple
        state = (
            taxi_row, taxi_col,                    # Taxi position
            self.stations[0][0], self.stations[0][1],  # Station R
            self.stations[1][0], self.stations[1][1],  # Station G
            self.stations[2][0], self.stations[2][1],  # Station Y
            self.stations[3][0], self.stations[3][1],  # Station B
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,  # Obstacles
            passenger_visible, destination_visible  # Visibility flags
        )
        return state
    
    def render(self):
        """Render the environment to the console"""
        clear_output(wait=True)
        
        # Create empty grid
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # Place stations
        for (x, y), label in zip(self.stations, self.STATION_LABELS):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[x][y] = label
        
        # Place passenger
        pass_row, pass_col = self.passenger_loc
        if not self.passenger_picked_up and 0 <= pass_row < self.grid_size and 0 <= pass_col < self.grid_size:
            grid[pass_row][pass_col] = 'P'
        
        # Place destination
        dest_row, dest_col = self.destination
        if 0 <= dest_row < self.grid_size and 0 <= dest_col < self.grid_size:
            grid[dest_row][dest_col] = 'D'
        
        # Place obstacles
        for row, col in self.obstacles:
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                grid[row][col] = "X"
        
        # Place taxi
        taxi_row, taxi_col = self.taxi_pos
        if 0 <= taxi_row < self.grid_size and 0 <= taxi_col < self.grid_size:
            taxi_symbol = 'T' if not self.passenger_picked_up else 'T+P'
            grid[taxi_row][taxi_col] = taxi_symbol
        
        # Print state information
        print(f"\nTaxi Position: ({taxi_row}, {taxi_col})")
        print(f"Passenger {'in taxi' if self.passenger_picked_up else f'at {self.passenger_loc}'}")
        print(f"Destination: {self.destination}")
        print(f"Fuel Left: {self.current_fuel}/{self.fuel_limit}")
        print(f"Grid Size: {self.grid_size}x{self.grid_size}, Obstacles: {len(self.obstacles)}")
        print("\nEnvironment:")
        
        # Print grid
        for row in grid:
            print(" ".join(row))
        print()


def run_agent(agent_file: str, render: bool = False, num_episodes: int = 1):
    """
    Run an agent from a file against the environment
    
    Args:
        agent_file: Path to agent file
        render: Whether to render the environment
        num_episodes: Number of episodes to run
    """
    # Import agent module
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)
    
    # Track metrics across episodes
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        # Create environment
        env = TrainingTaxiEnv(
            max_fuel=100,
            n=np.random.randint(5, 11),
            # difficulty='hard'
        )
        
        # Reset environment
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        # Extract initial position
        taxi_row, taxi_col = obs[0], obs[1]
        
        # Initial render
        if render:
            env.render()
            time.sleep(0.5)
        
        # Episode loop
        while not done:
            # Get action from agent
            action = student_agent.get_action(obs)
            
            # Take step in environment
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print step information
            if render:
                print(f"Step {step_count}: Action={TrainingTaxiEnv.ACTION_NAMES[action]}, Reward={reward:.1f}")
                env.render()
                time.sleep(0.1)
        
        # Episode summary
        print(f"Episode {episode+1}/{num_episodes}:")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Success: {'Yes' if total_reward > 0 else 'No'}")
        print()
        
        # Record metrics
        total_rewards.append(total_reward)
        total_steps.append(step_count)
    
    # Final summary
    print("=" * 40)
    print(f"Completed {num_episodes} episodes")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(total_steps):.1f}")
    print(f"Success Rate: {sum(r > 0 for r in total_rewards)/num_episodes:.2%}")
    
    return total_rewards


if __name__ == "__main__":
    # Run agent with rendering for 3 episodes
    results = run_agent("student_agent.py", render=True, num_episodes=50)
    print(f"Final Results: {results}")