import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    """
    Policy Network (Actor)
    Maps state to a probability distribution over actions.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_fc = nn.Linear(256, action_dim)
        self.log_std_fc = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Use reparameterization trick for backpropagation
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_deterministic_action(self, state):
        """
        Gets the deterministic action for evaluation (the mean of the distribution).
        """
        mean, _ = self.forward(state)
        action = torch.tanh(mean) * self.max_action
        return action


class Critic(nn.Module):
    """
    Q-Network (Critic)
    Maps (state, action) pairs to a Q-value.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Critic 2 (for Twin-Delayed DDPG / SAC)
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2