import numpy as np
from collections import defaultdict

class AdvancedTaxiAgent:
    def __init__(self, nA=6, alpha=0.1, gamma=0.99, beta=0.9, c1=2.0, c2=1.0,
                 get_epsilon=lambda i: max(0.01, 0.9 * (0.99 ** i)),
                 get_alpha=None, get_gamma=None, get_beta=None,
                 get_c1=None, get_c2=None):
        """
        Initialize an advanced agent with path memory for the taxi environment.
        
        Parameters
        ----------
        nA : int
            Number of actions available to the agent
        alpha : float
            Default learning rate
        gamma : float
            Default discount rate
        beta : float
            Default decay rate of path memory
        c1 : float
            Default weight of value function in stochastic action distribution
        c2 : float
            Default (inverse) weight of path memory in stochastic action distribution
        get_epsilon : function
            Function to calculate epsilon based on episode number
        get_alpha, get_gamma, get_beta, get_c1, get_c2 : function or None
            Functions to update respective parameters based on episode number
        """
        # Store initial parameter values
        self.alpha_init = alpha
        self.gamma_init = gamma
        self.beta_init = beta
        self.c1_init = c1
        self.c2_init = c2
        
        # Set parameter update functions
        self.get_epsilon = get_epsilon
        self.get_alpha = (lambda i: self.alpha_init) if get_alpha is None else get_alpha
        self.get_gamma = (lambda i: self.gamma_init) if get_gamma is None else get_gamma
        self.get_beta = (lambda i: self.beta_init) if get_beta is None else get_beta
        self.get_c1 = (lambda i: self.c1_init) if get_c1 is None else get_c1
        self.get_c2 = (lambda i: self.c2_init) if get_c2 is None else get_c2
        
        # Action space size
        self.nA = nA
        
        # Initialize action-value table and path memory table
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.recent = defaultdict(lambda: np.zeros(self.nA))
        
        # Current parameter values
        self.epsilon = self.get_epsilon(0)
        self.alpha = self.get_alpha(0)
        self.gamma = self.get_gamma(0)
        self.beta = self.get_beta(0)
        self.c1 = self.get_c1(0)
        self.c2 = self.get_c2(0)
        
        # Episode counter
        self.i_episode = 0
        
    def softmax(self, x):
        """
        Compute softmax values for each set of scores in x.
        
        Parameters
        ----------
        x : array-like
            Input array
            
        Returns
        -------
        array-like
            Softmax probabilities
        """
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()
    
    def get_action(self, state):
        """
        Select an action using the modified epsilon-greedy policy.
        
        Parameters
        ----------
        state : tuple
            Current state of the environment
            
        Returns
        -------
        int
            Selected action
        """
        # Convert state to a hashable tuple if needed
        state = tuple(state)
        
        # If we have no record of this state, choose a random action
        if state not in self.Q:
            return np.random.choice(self.nA)
        
        # Extract Q-values and path memory for this state
        q_values = np.array(self.Q[state])
        path_memory = np.array(self.recent[state])
        
        # Determine whether to explore or exploit
        if np.random.random() < self.epsilon:
            # Exploration: Use a weighted distribution based on Q-values and path memory
            preference = self.c1 * q_values - self.c2 * path_memory
            action_probs = self.softmax(preference)
            return np.random.choice(self.nA, p=action_probs)
        else:
            # Exploitation: Choose the action with the highest Q-value
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge using Q-learning and track action frequency.
        
        Parameters
        ----------
        state : tuple
            Previous state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : tuple
            Current state
        done : bool
            Whether the episode is complete
        """
        # Convert states to hashable tuples
        state = tuple(state)
        next_state = tuple(next_state)
        
        # Q-learning update
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action] * (1 - int(done))
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        
        # Update path memory by incrementing count for the chosen action
        self.recent[state][action] += 1
        
        # If episode is complete, update parameters and decay path memory
        if done:
            # Decay path memory from current episode
            for s in self.recent:
                self.recent[s] = self.recent[s] * self.beta
            
            # Update parameters for the next episode
            self.i_episode += 1
            self.epsilon = self.get_epsilon(self.i_episode)
            self.alpha = self.get_alpha(self.i_episode)
            self.gamma = self.get_gamma(self.i_episode)
            self.beta = self.get_beta(self.i_episode)
            self.c1 = self.get_c1(self.i_episode)
            self.c2 = self.get_c2(self.i_episode)
    
    def train(self, env, num_episodes=5000, max_steps=1000, render=False):
        """
        Train the agent on the given environment.
        
        Parameters
        ----------
        env : gym.Env
            The training environment
        num_episodes : int
            Number of training episodes
        max_steps : int
            Maximum steps per episode
        render : bool
            Whether to render the environment during training
            
        Returns
        -------
        list
            Episode rewards
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Reset environment
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Select and take action
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                
                # Update Q-values and path memory
                self.update(state, action, reward, next_state, done)
                
                # Update state and total reward
                state = next_state
                total_reward += reward
                
                # Render if specified
                if render and step % 10 == 0:  # Render every 10 steps to speed up training
                    taxi_row, taxi_col = state[0], state[1]
                    env.render_env((taxi_row, taxi_col), action=action, 
                                  step=step, fuel=env.current_fuel)
                
                if done:
                    break
            
            # Track episode reward
            episode_rewards.append(total_reward)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, " 
                      f"Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Beta: {self.beta:.4f}")
        
        return episode_rewards
    
    def save_policy(self, filename):
        """
        Save the Q-table and path memory to a file.
        
        Parameters
        ----------
        filename : str
            Filename to save to
        """
        # Convert to regular dictionaries for saving
        q_dict = {str(state): values.tolist() for state, values in self.Q.items()}
        recent_dict = {str(state): values.tolist() for state, values in self.recent.items()}
        
        # Save parameters and tables
        save_data = {
            'Q': q_dict,
            'recent': recent_dict,
            'params': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'beta': self.beta,
                'c1': self.c1,
                'c2': self.c2,
                'epsilon': self.epsilon,
                'i_episode': self.i_episode
            }
        }
        
        np.save(filename, save_data, allow_pickle=True)
        print(f"Policy saved to {filename}")
    
    def load_policy(self, filename):
        """
        Load the Q-table and path memory from a file.
        
        Parameters
        ----------
        filename : str
            Filename to load from
            
        Returns
        -------
        bool
            Whether the load was successful
        """
        try:
            saved_data = np.load(filename, allow_pickle=True).item()
            
            # Load Q-values
            for state_str, values in saved_data['Q'].items():
                state = eval(state_str)
                self.Q[state] = np.array(values)
            
            # Load path memory
            for state_str, values in saved_data['recent'].items():
                state = eval(state_str)
                self.recent[state] = np.array(values)
            
            # Load parameters
            params = saved_data['params']
            self.alpha = params['alpha']
            self.gamma = params['gamma']
            self.beta = params['beta']
            self.c1 = params['c1']
            self.c2 = params['c2']
            self.epsilon = params['epsilon']
            self.i_episode = params['i_episode']
            
            print(f"Policy loaded from {filename}")
            return True
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to load policy: {e}")
            return False

# For use with SimpleTaxiEnv's run_agent function
agent = AdvancedTaxiAgent()

def get_action(state):
    """
    Function to be used by the environment runner.
    
    Parameters
    ----------
    state : tuple
        Current state
        
    Returns
    -------
    int
        Selected action
    """
    return agent.get_action(state)

# Example usage
if __name__ == "__main__":
    try:
        from simple_custom_taxi_env import SimpleTaxiEnv
    except ImportError:
        print("Unable to import SimpleTaxiEnv. Make sure the module is available.")
        exit(1)
    
    # Create environment
    env = SimpleTaxiEnv(fuel_limit=5000)
    
    # Create agent with custom parameters
    agent = AdvancedTaxiAgent(
        nA=6,
        alpha=0.1,
        gamma=0.99,
        beta=0.9,
        c1=2.0,
        c2=1.0,
        get_epsilon=lambda i: max(0.01, 1.0 * (0.9995 ** i))
    )
    
    # Try to load a pre-trained policy
    if not agent.load_policy("advanced_taxi_policy.npy"):
        print("No pre-trained policy found. Starting fresh training.")
    
    # Train the agent
    rewards = agent.train(env, num_episodes=50000, max_steps=5000, render=False)
    
    # Save the trained policy
    agent.save_policy("advanced_taxi_policy.npy")
    
    # Evaluate the trained agent
    print("\nEvaluating the trained agent...")
    
    # Set epsilon to a small value for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.01
    
    # Run evaluation episodes
    eval_episodes = 5
    eval_rewards = []
    
    for episode in range(eval_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
            step += 1
            
            # Visualize
            taxi_row, taxi_col = state[0], state[1]
            env.render_env((taxi_row, taxi_col), action=action, 
                          step=step, fuel=env.current_fuel)
            
            # Sleep to make visualization readable
            import time
            time.sleep(0.1)
        
        eval_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    print(f"\nAverage Evaluation Reward: {np.mean(eval_rewards):.2f}")