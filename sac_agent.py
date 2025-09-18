import torch
import torch.optim as optim
import torch.nn.functional as F

from models import Actor, Critic

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Actor Network
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic Networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic temperature tuning
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):
        """
        Selects an action from the policy.
        
        Args:
            state: The current state.
            evaluate (bool): If True, returns a deterministic action. 
                             If False, returns a stochastic action.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            action = self.actor.get_deterministic_action(state)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Update Critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target, q2_target = self.critic_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target)
            soft_q_target = reward + (1 - done) * self.gamma * (q_target - self.alpha * next_log_prob)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, soft_q_target) + F.mse_loss(q2, soft_q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor and Alpha
        for p in self.critic.parameters(): p.requires_grad = False
        pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.critic.parameters(): p.requires_grad = True

        # Update Alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)