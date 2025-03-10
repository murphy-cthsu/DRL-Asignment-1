import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

# Import the SimpleTaxiEnv from your file
from simple_custom_taxi_env import SimpleTaxiEnv

# Import our agent
from Q_agent import TaxiAgent

def train_agent(env, agent, num_episodes=50000, max_steps=1000, 
                print_every=100, plot_every=100, render=False):
    """
    Train an agent on a given environment
    
    Parameters:
    -----------
    env : gym.Env
        The environment to train on
    agent : TaxiAgent
        The agent to train
    num_episodes : int
        Number of episodes to train for
    max_steps : int
        Maximum steps per episode
    print_every : int
        How often to print training statistics
    plot_every : int
        How often to update the plot
    render : bool
        Whether to render the environment during training
        
    Returns:
    --------
    tuple
        (scores, avg_scores) - Lists of scores and average scores over time
    """
    # Keep track of scores and average scores
    scores = []
    avg_scores = []
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Render if specified
            if render:
                taxi_row, taxi_col = state[0], state[1]
                env.render_env((taxi_row, taxi_col), action=action, 
                              step=step, fuel=env.current_fuel)
                sleep(0.1)
            
            if done:
                break
        
        # Record score
        scores.append(total_reward)
        avg_scores.append(np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores))
        
        # Print progress
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}/{num_episodes}, " 
                  f"Score: {total_reward:.2f}, "
                  f"Average Score (last 100): {avg_scores[-1]:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
        
        # Plot progress
        if (episode + 1) % plot_every == 0:
            plot_training_progress(scores, avg_scores, episode + 1)
    
    # Final plot
    plot_training_progress(scores, avg_scores, num_episodes, final=True)
    
    return scores, avg_scores

def plot_training_progress(scores, avg_scores, episode, final=False):
    """Plot the training progress"""
    clear_output(wait=True)
    plt.figure(figsize=(12, 6))
    
    # Plot episode scores
    plt.plot(scores, alpha=0.6, label='Episode Score')
    
    # Plot average scores
    plt.plot(avg_scores, color='red', label='Average Score (last 100 episodes)')
    
    plt.axhline(y=50, color='green', linestyle='--', alpha=0.3, 
                label='Success threshold')
    
    plt.title(f'Training Progress (Episode {episode})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    if final:
        plt.savefig('taxi_training_progress.png')
    
    plt.show()

def evaluate_agent(env, agent, num_episodes=10, render=True):
    """
    Evaluate a trained agent
    
    Parameters:
    -----------
    env : gym.Env
        The environment to evaluate on
    agent : TaxiAgent
        The trained agent to evaluate
    num_episodes : int
        Number of episodes to evaluate for
    render : bool
        Whether to render the environment during evaluation
        
    Returns:
    --------
    float
        Average evaluation score
    """
    # Set agent to evaluation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    eval_scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Select action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            step += 1
            
            # Render if specified
            if render:
                taxi_row, taxi_col = state[0], state[1]
                env.render_env((taxi_row, taxi_col), action=action, 
                              step=step, fuel=env.current_fuel)
                sleep(0.2)  # Slow down the rendering for better visualization
        
        eval_scores.append(total_reward)
        print(f"Evaluation Episode {episode + 1}: Score = {total_reward:.2f}")
    
    # Restore agent's original epsilon
    agent.epsilon = original_epsilon
    
    avg_score = np.mean(eval_scores)
    print(f"\nEvaluation complete! Average Score: {avg_score:.2f}")
    return avg_score

if __name__ == "__main__":
    # Create environment
    env = SimpleTaxiEnv(fuel_limit=5000)
    
    # Create agent
    agent = TaxiAgent(nA=6, learning_rate=0.5, gamma=0.99, epsilon=1.0, 
                     epsilon_min=0.1, epsilon_decay=0.999)
    
    # Train agent
    print("Starting training...")
    scores, avg_scores = train_agent(env, agent, num_episodes=50000, max_steps=1000, 
                                    print_every=100, plot_every=500)
    
    # Save the trained Q-values
    q_dict = {str(state): values.tolist() for state, values in agent.Q.items()}
    np.save("taxi_agent_qvalues.npy", q_dict)
    print("Training completed and Q-values saved!")
    
    # Evaluate the trained agent
    print("\nEvaluating the trained agent...")
    avg_eval_score = evaluate_agent(env, agent, num_episodes=5, render=True)