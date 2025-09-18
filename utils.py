import matplotlib.pyplot as plt

def plot_rewards(prey_rewards, adversary_rewards):
    """Plots the rewards for prey and adversaries."""
    plt.figure(figsize=(12, 6))
    plt.plot(prey_rewards, label='Prey (Good Agent) Reward')
    plt.plot(adversary_rewards, label='Adversary (Bad Agent) Avg Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('SAC Training in Simple Tag')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_rewards.png') # Save the plot to a file
    plt.show()