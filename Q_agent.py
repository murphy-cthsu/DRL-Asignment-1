import numpy as np
from collections import defaultdict

class TaxiAgent:
    def __init__(self, nA=6, learning_rate=0.1, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize TaxiAgent for the custom Taxi environment.
        
        Parameters:
        -----------
        nA : int
            Number of possible actions (default: 6 for the taxi environment)
        learning_rate : float
            Learning rate (alpha) for Q-learning updates
        gamma : float
            Discount factor for future rewards
        epsilon : float
            Starting exploration rate
        epsilon_min : float
            Minimum exploration rate
        epsilon_decay : float
            Rate at which epsilon decreases over time
        """
        # Action space size
        self.nA = nA
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table as a defaultdict
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        # Keep track of training episodes
        self.episode_count = 0
    
    def get_action(self, state):
        """
        Select an action using an epsilon-greedy policy.
        
        Parameters:
        -----------
        state : tuple
            The current state of the environment
            
        Returns:
        --------
        int
            The selected action
        """
        # Convert state tuple to a hashable format if needed
        state = tuple(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: select a random action
            return np.random.randint(self.nA)
        else:
            # Exploit: select the action with the highest Q-value
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning algorithm.
        
        Parameters:
        -----------
        state : tuple
            The previous state
        action : int
            The action taken
        reward : float
            The reward received
        next_state : tuple
            The resulting state
        done : bool
            Whether the episode is complete
        """
        # Convert state tuples to hashable format
        state = tuple(state)
        next_state = tuple(next_state)
        
        # Q-learning update
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action] * (1 - int(done))
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error
        
        # Decay epsilon if episode is done
        if done:
            self.episode_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes=1000, max_steps=1000, render=False):
        """
        Train the agent on the given environment.
        
        Parameters:
        -----------
        env : gym.Env
            The training environment
        num_episodes : int
            Number of training episodes
        max_steps : int
            Maximum steps per episode
        render : bool
            Whether to render the environment during training
        
        Returns:
        --------
        list
            Episode rewards
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Reset the environment
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Select action
                action = self.get_action(state)
                
                # Take action and observe next state and reward
                next_state, reward, done, _, _ = env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Render if specified
                if render:
                    taxi_row, taxi_col = state[0], state[1]
                    env.render_env((taxi_row, taxi_col), action=action, 
                                  step=step, fuel=env.current_fuel)
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, Average Reward (last 100): {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        return episode_rewards
                
    def save_policy(self, filename):
        """
        Save the Q-table to a file.
        
        Parameters:
        -----------
        filename : str
            The filename to save the Q-table to
        """
        # Convert Q-table to a regular dictionary for saving
        q_dict = {str(state): values.tolist() for state, values in self.Q.items()}
        np.save(filename, q_dict, allow_pickle=True)
    
    def load_policy(self, filename):
        """
        Load a Q-table from a file.
        
        Parameters:
        -----------
        filename : str
            The filename to load the Q-table from
        """
        q_dict = np.load(filename, allow_pickle=True).item()
        
        # Convert keys back to tuples
        for state_str, values in q_dict.items():
            state = eval(state_str)
            self.Q[state] = np.array(values)

# For usage with the given SimpleTaxiEnv
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

# Example usage:
if __name__ == "__main__":
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Create environment
    env = SimpleTaxiEnv(fuel_limit=5000)
    
    # Create and train agent
    agent = TaxiAgent(nA=6, learning_rate=0.1, gamma=0.99, epsilon=1.0, 
                     epsilon_min=0.01, epsilon_decay=0.995)
    
    # Train the agent
    rewards = agent.train(env, num_episodes=5000, max_steps=1000, render=False)
    
    # Save the trained agent
    agent.save_policy("taxi_agent_policy.npy")
    
    # Test the agent
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        # Visualize
        taxi_row, taxi_col = state[0], state[1]
        env.render_env((taxi_row, taxi_col), action=action, 
                      step=None, fuel=env.current_fuel)
    
    print(f"Test episode completed with total reward: {total_reward}")