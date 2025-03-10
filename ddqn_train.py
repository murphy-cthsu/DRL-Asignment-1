import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

# Import your custom environment
from simple_custom_taxi_env import SimpleTaxiEnv
from ddqn import DDQNAgent

def train_agent(env, agent, num_episodes=1000, max_steps=1000, render_interval=100, save_path="models", 
                save_interval=100, eval_interval=100, num_eval_episodes=10, early_stop_score=200):
    """
    Train the DDQN agent on the custom Taxi environment
    
    Args:
        env: The environment to train on
        agent: The DDQN agent
        num_episodes: Total number of episodes to train for
        max_steps: Maximum steps per episode
        render_interval: How often to render the environment (episodes)
        save_path: Directory to save model checkpoints
        save_interval: How often to save model checkpoints (episodes)
        eval_interval: How often to evaluate the agent (episodes)
        num_eval_episodes: Number of episodes to evaluate the agent on
        early_stop_score: Score threshold for early stopping
    """
    # Create directory for saving models if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Lists to keep track of rewards and losses
    episode_rewards = []
    evaluation_scores = []
    avg_losses = []
    
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Render environment periodically
            if episode % render_interval == 0 and step == 0:
                taxi_row, taxi_col = int(state[0]), int(state[1])
                env.render_env((taxi_row, taxi_col), action=None, step=step, fuel=env.current_fuel)
                time.sleep(0.3)
            
            # Get action
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            
            
            if done:
                break
        
        # Save episode reward
        episode_rewards.append(episode_reward)
        
        # Save average loss for this episode
        if episode_losses:
            avg_losses.append(np.mean(episode_losses))
        
        # Print info
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save(f"{save_path}/ddqn_agent_episode_{episode+1}.pt")
        
        # Evaluate agent periodically
        if (episode + 1) % eval_interval == 0:
            eval_score = evaluate_agent(env, agent, num_episodes=num_eval_episodes)
            evaluation_scores.append(eval_score)
            print(f"Evaluation Score (over {num_eval_episodes} episodes): {eval_score:.2f}")
            
            # Early stopping based on evaluation score
            if eval_score >= early_stop_score:
                print(f"Early stopping at episode {episode+1} with eval score {eval_score:.2f} >= {early_stop_score}")
                agent.save(f"{save_path}/ddqn_agent_final.pt")
                break
    
    # Save final model
    agent.save(f"{save_path}/ddqn_agent_final.pt")
    
    # Plot training results
    plot_training_results(episode_rewards, avg_losses, evaluation_scores, eval_interval)
    
    return agent

def evaluate_agent(env, agent, num_episodes=10, max_steps=1000, render=False):
    """
    Evaluate the agent's performance
    
    Args:
        env: The environment to evaluate on
        agent: The DDQN agent
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment during evaluation
    
    Returns:
        float: Average reward over all evaluation episodes
    """
    total_rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.01  # Use minimal exploration during evaluation
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            episode_reward += reward
            state = next_state
            
            if render:
                taxi_row, taxi_col = state[0], state[1]
                env.render_env((taxi_row, taxi_col), action=action, step=step, fuel=env.current_fuel)
                time.sleep(0.1)
                
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return np.mean(total_rewards)

def plot_training_results(rewards, losses, eval_scores, eval_interval):
    """
    Plot training results
    
    Args:
        rewards: List of episode rewards
        losses: List of episode losses
        eval_scores: List of evaluation scores
        eval_interval: Interval at which evaluations were performed
    """
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.plot(np.arange(len(rewards)), np.convolve(rewards, np.ones(10)/10, mode='same'), 'r')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot episode losses
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot evaluation scores
    plt.subplot(3, 1, 3)
    eval_episodes = np.arange(1, len(eval_scores) + 1) * eval_interval
    plt.plot(eval_episodes, eval_scores)
    plt.title('Evaluation Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    # Create environment
    env_config = {
        "fuel_limit": 5000
    }
    env = SimpleTaxiEnv(**env_config)
    
    # Create agent
    state_size = 16  # Based on your environment's observation space
    action_size = 6  # Based on your environment's action space
    
    agent = DDQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        tau=0.001,
        batch_size=64
    )
    
    # Train agent
    trained_agent = train_agent(
        env=env,
        agent=agent,
        num_episodes=1000,
        max_steps=500,
        render_interval=200,
        save_interval=100,
        eval_interval=50,
        num_eval_episodes=10,
        early_stop_score=50
    )
    
    # Evaluate final agent performance
    print("Evaluating final agent performance...")
    final_score = evaluate_agent(env, trained_agent, num_episodes=50, render=True)
    print(f"Final evaluation score (over 50 episodes): {final_score:.2f}")