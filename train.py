# train.py
import numpy as np # Import numpy
from tqdm import tqdm
from pettingzoo.mpe import simple_tag_v3
import torch

# Import project components
import config
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from utils import plot_rewards
from pragmatic_wrapper import PragmaticWrapper

def train():
    """Main training function."""
    print(f"Using device: {config.DEVICE}")

    # --- Environment Setup ---
    env = simple_tag_v3.parallel_env(**config.ENV_CONFIG)
    env = PragmaticWrapper(env)
    
    adversary_ids = [agent for agent in env.possible_agents if 'adversary' in agent]
    prey_ids = [agent for agent in env.possible_agents if 'agent' in agent]
    
    # Adversary Agent (shared policy)
    adv_obs_space = env.observation_space(adversary_ids[0])
    adv_action_space = env.action_space(adversary_ids[0])
    adversary_agent = SACAgent(
        state_dim=adv_obs_space.shape[0],
        action_dim=adv_action_space.shape[0],
        max_action=1.0,
        device=config.DEVICE,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        tau=config.TAU
    )
    adversary_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY)

    # Prey Agent
    prey_obs_space = env.observation_space(prey_ids[0])
    prey_action_space = env.action_space(prey_ids[0])
    prey_agent = SACAgent(
        state_dim=prey_obs_space.shape[0],
        action_dim=prey_action_space.shape[0],
        max_action=1.0,
        device=config.DEVICE,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        tau=config.TAU
    )
    prey_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY)

    # --- Logging ---
    episode_rewards_prey = []
    episode_rewards_adversaries = [[] for _ in range(len(adversary_ids))]

    # --- Training Loop ---
    for episode in tqdm(range(config.NUM_EPISODES)):
        obs, _ = env.reset()
        episode_reward_prey = 0
        episode_reward_adversaries_per_episode = [0] * len(adversary_ids)

        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Store the original [-1, 1] actions for the replay buffer
            original_actions = {}
            # Store the rescaled [0, 1] actions for the environment
            env_actions = {}

            # --- Action Selection and Rescaling ---
            for agent_id in adversary_ids:
                if agent_id in obs:
                    # 1. Get original [-1, 1] action
                    act = adversary_agent.select_action(obs[agent_id])
                    original_actions[agent_id] = act
                    # 2. Rescale to [0, 1] for the environment
                    env_actions[agent_id] = (act + 1.0) / 2.0
            
            for agent_id in prey_ids:
                 if agent_id in obs:
                    # 1. Get original [-1, 1] action
                    # act = prey_agent.select_action(obs[agent_id])
                    act = np.random.uniform(-1, 1, size=prey_action_space.shape[0])
                    original_actions[agent_id] = act
                    # 2. Rescale to [0, 1] for the environment
                    env_actions[agent_id] = ((act + 1.0) / 2.0).astype(np.float32)

            # --- Step the Environment with Rescaled Actions ---
            next_obs, rewards, terminations, truncations, _ = env.step(env_actions)
            
            # --- Store Experience with Original Actions ---
            for agent_id in obs.keys():
                is_done = terminations[agent_id] or truncations[agent_id]
                
                if agent_id in next_obs:
                    if 'adversary' in agent_id:
                        # Push the ORIGINAL [-1, 1] action to the buffer
                        adversary_buffer.push(obs[agent_id], original_actions[agent_id], rewards[agent_id], next_obs[agent_id], is_done)
                        # Get the index of the adversary to store the reward
                        adversary_index = adversary_ids.index(agent_id)
                        episode_reward_adversaries_per_episode[adversary_index] += rewards[agent_id]
                    else:
                        # Push the ORIGINAL [-1, 1] action to the buffer
                        prey_buffer.push(obs[agent_id], original_actions[agent_id], rewards[agent_id], next_obs[agent_id], is_done)
                        episode_reward_prey += rewards[agent_id]

            obs = next_obs
            
            adversary_agent.update(adversary_buffer, config.BATCH_SIZE)
            # prey_agent.update(prey_buffer, config.BATCH_SIZE)

            if not obs:
                break

        episode_rewards_prey.append(episode_reward_prey)
        for i in range(len(adversary_ids)):
            episode_rewards_adversaries[i].append(episode_reward_adversaries_per_episode[i])

    env.close()

    # --- Save Models ---
    torch.save(adversary_agent.actor.state_dict(), 'models/sac_simple_tag/adversary_actor.pth')
    torch.save(adversary_agent.critic.state_dict(), 'models/sac_simple_tag/adversary_critic.pth')
    torch.save(prey_agent.actor.state_dict(), 'models/sac_simple_tag/prey_actor.pth')
    torch.save(prey_agent.critic.state_dict(), 'models/sac_simple_tag/prey_critic.pth')
    print("Models saved successfully!")

    # --- Plotting Results ---
    plot_rewards(episode_rewards_prey, episode_rewards_adversaries)

if __name__ == "__main__":
    train()