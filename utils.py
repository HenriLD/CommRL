import matplotlib.pyplot as plt

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