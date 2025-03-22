from typing import Optional, List, Dict, Union, Tuple, Any
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from my_env import TrainingTaxiEnv
import matplotlib.pyplot as plt
import argparse
import os
import json
import yaml
from collections import deque

# Import our refactored state management
from state import TaxiStateTracker, ACTION_COUNT, enhance_reward

# Evaluation window size
EVAL_WINDOW = 100


class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize training configuration with default values or from dictionary
        
        Args:
            config_dict: Dictionary of configuration values (optional)
        """
        # Training parameters
        self.episodes = 8000
        self.learning_rate = 0.01
        self.discount_factor = 0.99
        self.entropy_coefficient = 0.01
        self.weight_decay = 1e-5
        self.grad_clip = 1.0
        
        # Environment parameters
        self.min_grid_size = 5
        self.max_grid_size = 11
        self.fuel_limit = 1000
        self.difficulty_progression = True  # Whether to increase difficulty over time
        
        # Model parameters
        self.hidden_sizes = [16]
        self.dropout_rate = 0
        
        # Output parameters
        self.model_save_path = 'trained_models/policy_model.pth'
        self.plot_dir = 'training_plots'
        self.checkpoint_path = None
        self.save_frequency = 100
        
        # Optimizer parameters
        self.scheduler_patience = 10
        self.scheduler_factor = 0.5
        self.scheduler_threshold = 0.01
        
        # If config dictionary is provided, update values
        if config_dict:
            self.__dict__.update(config_dict)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'TrainingConfig':
        """
        Load configuration from a file
        
        Args:
            filepath: Path to configuration file (json or yaml)
            
        Returns:
            TrainingConfig instance
        """
        if not os.path.exists(filepath):
            print(f"Configuration file {filepath} not found. Using default configuration.")
            return cls()
        
        try:
            # Determine file type by extension
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)
            elif filepath.endswith(('.yaml', '.yml')):
                with open(filepath, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                print(f"Unsupported file format: {filepath}. Using default configuration.")
                return cls()
            
            return cls(config_dict)
        
        except Exception as e:
            print(f"Error loading configuration from {filepath}: {e}")
            print("Using default configuration.")
            return cls()
    
    def save(self, filepath: str):
        """
        Save configuration to a file
        
        Args:
            filepath: Path to save configuration
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        config_dict = self.__dict__
        
        try:
            # Determine file type by extension
            if filepath.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(config_dict, f, indent=4)
            elif filepath.endswith(('.yaml', '.yml')):
                with open(filepath, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                print(f"Unsupported file format: {filepath}. Using JSON format.")
                with open(f"{filepath}.json", 'w') as f:
                    json.dump(config_dict, f, indent=4)
        
        except Exception as e:
            print(f"Error saving configuration to {filepath}: {e}")


class PolicyModel(nn.Module):
    """Enhanced policy network with improved architecture and utilities"""
    
    def __init__(self, input_dim: int, output_dim: int, config: TrainingConfig):
        """
        Initialize the enhanced policy network
        
        Args:
            input_dim: Dimension of state input
            output_dim: Dimension of action output
            config: Training configuration
        """
        super(PolicyModel, self).__init__()
        
        # Build network layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_dim
        
        for size in config.hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            # Add dropout for regularization
            layers.append(nn.Dropout(config.dropout_rate))
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


class TrainingMetrics:
    """Class to track and visualize training metrics"""
    
    def __init__(self, window_size: int = EVAL_WINDOW):
        """Initialize metrics tracking"""
        self.window_size = window_size
        self.rewards = []
        self.successes = []
        self.step_counts = []
        self.losses = []
        self.unique_states = set()
    
    def add_episode_result(self, reward: float, success: bool, steps: int, 
                          loss: float, states_visited: set):
        """Add results from a single episode"""
        self.rewards.append(reward)
        self.successes.append(int(success))
        self.step_counts.append(steps)
        self.losses.append(loss)
        self.unique_states.update(states_visited)
    
    def get_recent_metrics(self) -> Dict[str, float]:
        """Get metrics averaged over recent window"""
        window = min(self.window_size, len(self.rewards))
        
        return {
            'avg_reward': np.mean(self.rewards[-window:]),
            'success_rate': np.mean(self.successes[-window:]),
            'avg_steps': np.mean(self.step_counts[-window:]),
            'avg_loss': np.mean(self.losses[-window:]) if self.losses else 0,
            'unique_states': len(self.unique_states)
        }
    
    def save_to_file(self, filepath: str):
        """Save metrics to file for later analysis"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'rewards': self.rewards,
            'successes': self.successes,
            'step_counts': self.step_counts,
            'losses': self.losses,
            'unique_states_count': len(self.unique_states)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def plot_training_progress(self, output_path: str = 'training_plots'):
        """Create and save plots of training progress"""
        os.makedirs(output_path, exist_ok=True)
        
        # Calculate moving average
        if len(self.rewards) >= self.window_size:
            moving_avg = np.convolve(
                self.rewards, 
                np.ones(self.window_size) / self.window_size,
                mode='valid'
            )
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.rewards, alpha=0.3, color='blue', label='Episode Reward')
            plt.plot(
                range(self.window_size - 1, len(self.rewards)),
                moving_avg, 
                color='red', 
                linewidth=2,
                label=f'{self.window_size}-Episode Average'
            )
            
            # Success rate as area plot
            if len(self.successes) >= self.window_size:
                success_rate = np.convolve(
                    self.successes,
                    np.ones(self.window_size) / self.window_size,
                    mode='valid'
                )
                plt.fill_between(
                    range(self.window_size - 1, len(self.successes)),
                    0, 
                    success_rate,
                    color='green',
                    alpha=0.2,
                    label='Success Rate'
                )
            
            plt.xlabel("Episodes")
            plt.ylabel("Total Reward")
            plt.title("Policy Gradient Training Progress")
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(f'{output_path}/training_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create step count plot
            plt.figure(figsize=(10, 6))
            if len(self.step_counts) >= self.window_size:
                steps_avg = np.convolve(
                    self.step_counts,
                    np.ones(self.window_size) / self.window_size,
                    mode='valid'
                )
                plt.plot(
                    range(self.window_size - 1, len(self.step_counts)),
                    steps_avg,
                    color='purple',
                    linewidth=2,
                    label=f'{self.window_size}-Episode Average Steps'
                )
            
            plt.xlabel("Episodes")
            plt.ylabel("Steps per Episode")
            plt.title("Episode Length During Training")
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Save step count plot
            plt.savefig(f'{output_path}/steps_curve.png', dpi=300, bbox_inches='tight')
            plt.close()


def train_policy_gradient(config: TrainingConfig):
    """
    Train a policy gradient agent on the taxi environment
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Create directories
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.plot_dir, 'training_config.yaml')
    config.save(config_path)
    
    # Initialize state tracker and metrics
    state_tracker = TaxiStateTracker()
    metrics = TrainingMetrics(window_size=EVAL_WINDOW)
    
    # Create enhanced policy network
    policy_model = PolicyModel(
        input_dim=state_tracker.state_dimension,
        output_dim=ACTION_COUNT,
        config=config
    )
    
    # Load checkpoint if provided
    if config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
        policy_model.load(config.checkpoint_path)
        print(f"Loaded checkpoint from {config.checkpoint_path}")
    
    # Setup optimizer with weight decay for regularization
    optimizer = optim.Adam(
        policy_model.parameters(), 
        lr=config.learning_rate
    )
    
    # Training loop with progress bar
    with tqdm(range(config.episodes)) as progress_bar:
        for episode in progress_bar:
            # Determine difficulty based on progression
            if config.difficulty_progression:
                difficulty = 'normal' if episode < config.episodes / 3 else 'hard'
            else:
                difficulty = 'hard'  # Always hard if no progression
                
            # Set up environment
            grid_size = np.random.randint(config.min_grid_size, config.max_grid_size)
            
            env = TrainingTaxiEnv(
                max_fuel=config.fuel_limit,
                n=grid_size,
                difficulty=difficulty
            )
            
            # Initialize episode variables
            log_probs = []
            rewards = []
            total_reward = 0
            
            # Reset environment and state tracker
            observation, _ = env.reset()
            state_tracker.initialize()
            state, info = state_tracker.process_observation(observation)
            metrics.unique_states.add(state)
            
            # Episode loop
            done = False
            success = False
            step_count = 0
            
            while not done:
                # Select action
                action_probs = policy_model(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                
                # Update state tracker with action
                state_tracker.update_after_action(action.item())
                
                # Take action in environment
                next_observation, reward, done, truncated, _ = env.step(action.item())
                next_state, next_info = state_tracker.process_observation(next_observation)
                metrics.unique_states.add(next_state)
                
                # Check for success
                if done and reward > 49:
                    success = True
                
                # Apply reward shaping
                shaped_reward = enhance_reward(reward, info, next_info)
                
                # Add shaped reward to episode totals
                total_reward += shaped_reward
                rewards.append(shaped_reward)
                
                # Update state for next step
                state = next_state
                info = next_info
                step_count += 1
            
            # Add episode results to metrics
            metrics.rewards.append(total_reward)
            metrics.successes.append(success)
            metrics.step_counts.append(step_count)
            
            # Calculate discounted returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + config.discount_factor * R
                returns.insert(0, R)
            
            # Normalize returns for stability
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Calculate policy loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            loss = torch.stack(policy_loss).sum()
            
            # Update metrics
            metrics.losses.append(loss.item())
            
            # Update policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Get recent metrics
            current_metrics = metrics.get_recent_metrics()
            
            # Update progress bar
            progress_bar.set_postfix({
                'reward': f"{current_metrics['avg_reward']:.1f}",
                'win rate': f"{current_metrics['success_rate']:.2f}",
                'step': f"{current_metrics['avg_steps']:.1f}",
                'state': current_metrics['unique_states']
            })
            
            # Periodically save model and plot progress
            if (episode + 1) % config.save_frequency == 0:
                # Plot progress
                plt.clf()
                plt.plot(metrics.rewards, alpha=0.5)
                if len(metrics.rewards) >= EVAL_WINDOW:
                    moving_avg = np.convolve(
                        metrics.rewards,
                        np.ones(EVAL_WINDOW) / EVAL_WINDOW,
                        mode='valid'
                    )
                    plt.plot(
                        range(EVAL_WINDOW - 1, len(metrics.rewards)),
                        moving_avg,
                        color='red'
                    )
                plt.xlabel("Episodes")
                plt.ylabel("Total Reward")
                plt.title("Training Progress")
                plt.grid()
                plt.savefig(f'{config.plot_dir}/training_curve.png')
                
                # Save model
                policy_model.save(config.model_save_path)
    
    # Final save
    policy_model.save(config.model_save_path)
    
    print(f"Training completed. Model saved to {config.model_save_path}")
    print(f"Final metrics: Success rate={current_metrics['success_rate']:.2f}, " 
          f"Avg reward={current_metrics['avg_reward']:.1f}")
    
    return policy_model, metrics


if __name__ == '__main__':
    # Parse command line to get configuration file path
    parser = argparse.ArgumentParser(description='Train a policy gradient agent for taxi navigation')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('-o', '--output', type=str, default='trained_models/policy_model.pth', help='Path to save the model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    args = parser.parse_args()

    if args.config:
        config = TrainingConfig.from_file(args.config)
    else:
        # Generate default configuration
        config = TrainingConfig()
        # Apply command line arguments
        config.model_save_path = args.output
        config.checkpoint_path = args.checkpoint
        config.learning_rate = args.lr
        config.discount_factor = args.gamma
        
        default_config_path = 'configs/default_config.yaml'
        os.makedirs(os.path.dirname(default_config_path), exist_ok=True)
        config.save(default_config_path)
        print(f"No configuration provided. Using default configuration saved to {default_config_path}")
    
    # Run training
    train_policy_gradient(config)