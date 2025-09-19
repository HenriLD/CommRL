import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
import torch
import os

def plot_rewards(prey_rewards, adversary_rewards):
    """
    Plots the rewards for prey and each adversary on separate subplots.
    
    Args:
        prey_rewards (list): A list of cumulative rewards for the prey agent per episode.
        adversary_rewards (list of lists): A list where each inner list contains the
                                           cumulative rewards for an adversary per episode.
    """
    num_adversaries = len(adversary_rewards)
    num_plots = 2
    
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
    
    # If there's only one plot, axs will not be an array, so we wrap it in a list
    if num_plots == 1:
        axs = [axs]

    # Prey Rewards Plot
    axs[0].plot(prey_rewards, label='Prey (Good Agent) Reward', color='g')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Cumulative Reward')
    axs[0].set_title('Prey Agent Rewards')
    axs[0].legend()
    axs[0].grid(True)

    # Adversary Rewards Plots
    axs[1].plot(adversary_rewards[0], label=f'Adversary Reward', color='r')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Cumulative Reward')
    axs[1].set_title(f'Adversary Rewards')
    axs[1].legend()
    axs[1].grid(True)

    # Overall Title and Layout
    fig.suptitle('SAC Training in Simple Tag', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    # Save and Show
    plt.savefig('training_rewards.png') # Save the plot to a file
    plt.show()

class ReplayBuffer:
    """An optimized FIFO experience replay buffer that uses NumPy arrays."""
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )

    def __len__(self):
        return self.size
    
def load_checkpoint(adversary_agent, prey_agent, timestamp=None, safe_mode=True, latest=False):
    """
    Loads a checkpoint for both adversary and prey agents.

    Can load the latest checkpoint or a specific one identified by a timestamp.
    Includes a safe mode for non-strict model loading.
    """
    base_model_path = os.path.join('models', 'sac_simple_tag')

    if timestamp is None and latest:
        # Find and load the latest checkpoint
        all_checkpoints = [d for d in os.listdir(base_model_path) if os.path.isdir(os.path.join(base_model_path, d))]
        if not all_checkpoints:
            print("No checkpoints found")
            return
        latest_checkpoint_dir_name = sorted(all_checkpoints)[-1]
        checkpoint_dir = os.path.join(base_model_path, latest_checkpoint_dir_name)
        print(f"Loading latest models from: {checkpoint_dir}")
    else:
        # Load a specific checkpoint
        checkpoint_dir = os.path.join(base_model_path, timestamp)
        if not os.path.isdir(checkpoint_dir):
            print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
            return
        print(f"Loading specific models from: {checkpoint_dir}")

    # Helper function for loading a single model
    def _load(model, file_path):
        if not os.path.exists(file_path):
            print(f"Warning: Model file not found at {file_path}")
            return

        if safe_mode:
            try:
                model.load_state_dict(torch.load(file_path))
                print(f"Loaded model from {file_path}")
            except RuntimeError:
                state_dict = torch.load(file_path)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded model from {file_path} with non-strict loading (safe mode).")
        else:
            model.load_state_dict(torch.load(file_path), strict=True)
            print(f"Loaded model from {file_path} with strict loading.")

    # Load all agent models
    _load(adversary_agent.actor, os.path.join(checkpoint_dir, 'adversary_actor.pth'))
    _load(adversary_agent.critic, os.path.join(checkpoint_dir, 'adversary_critic.pth'))
    _load(prey_agent.actor, os.path.join(checkpoint_dir, 'prey_actor.pth'))
    _load(prey_agent.critic, os.path.join(checkpoint_dir, 'prey_critic.pth'))