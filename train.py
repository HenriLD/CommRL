import numpy as np # Import numpy
from tqdm import tqdm
from pettingzoo.mpe import simple_tag_v3
import torch
from datetime import datetime
import os
import json

# Import project components
import config
from sac_agent import SACAgent
from utils import ReplayBuffer, load_checkpoint, plot_rewards
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
    adversary_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY, adv_obs_space.shape[0], adv_action_space.shape[0])

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
    prey_buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY, prey_obs_space.shape[0], prey_action_space.shape[0])

    # --- Checkpoint Directory Setup ---
    base_model_path = os.path.join('models', 'sac_simple_tag')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(base_model_path, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints for this run will be saved in: {checkpoint_dir}")

    if config.RESUME_TRAINING:
        load_checkpoint(adversary_agent, prey_agent, safe_mode=False, timestamp=config.TIME_STAMP)

    # --- Logging & Alternating Training Setup ---
    episode_rewards_prey = []
    episode_rewards_adversaries = [[] for _ in range(len(adversary_ids))]
    current_training_agent = config.INITIAL_TRAINING_AGENT

    # --- Training Loop ---
    for episode in tqdm(range(config.NUM_EPISODES)):
        obs, _ = env.reset()
        episode_reward_prey = 0
        episode_reward_adversaries_per_episode = [0] * len(adversary_ids)

        alternate_ticker = 0

        for step in range(config.MAX_STEPS_PER_EPISODE):
            original_actions = {}
            env_actions = {}

            # --- Action Selection and Rescaling ---
            for agent_id in adversary_ids:
                if agent_id in obs:
                    act = adversary_agent.select_action(obs[agent_id])
                    original_actions[agent_id] = act
                    env_actions[agent_id] = (act + 1.0) / 2.0

            for agent_id in prey_ids:
                 if agent_id in obs:
                    act = prey_agent.select_action(obs[agent_id])
                    original_actions[agent_id] = act
                    env_actions[agent_id] = ((act + 1.0) / 2.0)

            next_obs, rewards, terminations, truncations, _ = env.step(env_actions)

            # --- Store Experience ---
            for agent_id in obs.keys():
                is_done = terminations[agent_id] or truncations[agent_id]
                if agent_id in next_obs:
                    if 'adversary' in agent_id:
                        adversary_buffer.push(obs[agent_id], original_actions[agent_id], rewards[agent_id], next_obs[agent_id], is_done)
                        adversary_index = adversary_ids.index(agent_id)
                        episode_reward_adversaries_per_episode[adversary_index] += rewards[agent_id]
                    else:
                        prey_buffer.push(obs[agent_id], original_actions[agent_id], rewards[agent_id], next_obs[agent_id], is_done)
                        episode_reward_prey += rewards[agent_id]

            obs = next_obs

            # --- Agent Updates (with alternating logic) ---
            if config.ALTERNATING_TRAINING:
                if current_training_agent == 'adversary':
                    adversary_agent.update(adversary_buffer, config.BATCH_SIZE)
                    alternate_ticker += 1
                else: # current_training_agent == 'prey'
                    prey_agent.update(prey_buffer, config.BATCH_SIZE)
                    alternate_ticker += 5
            else:
                # Original behavior: update both agents every step
                adversary_agent.update(adversary_buffer, config.BATCH_SIZE)
                prey_agent.update(prey_buffer, config.BATCH_SIZE)

            if not obs:
                break

        # --- Switch agent group at the end of the episode if interval is reached ---
        if config.ALTERNATING_TRAINING and (alternate_ticker + 1) % config.TRAINING_INTERVAL == 0:
            current_training_agent = 'prey' if current_training_agent == 'adversary' else 'adversary'
            tqdm.write(f"\nEpisode interval reached. Switching training to: {current_training_agent.upper()}")

        episode_rewards_prey.append(episode_reward_prey)
        for i in range(len(adversary_ids)):
            episode_rewards_adversaries[i].append(episode_reward_adversaries_per_episode[i])


    env.close()

    # --- Save Models ---
    torch.save(adversary_agent.actor.state_dict(), os.path.join(checkpoint_dir, 'adversary_actor.pth'))
    torch.save(adversary_agent.critic.state_dict(), os.path.join(checkpoint_dir, 'adversary_critic.pth'))
    torch.save(prey_agent.actor.state_dict(), os.path.join(checkpoint_dir, 'prey_actor.pth'))
    torch.save(prey_agent.critic.state_dict(), os.path.join(checkpoint_dir, 'prey_critic.pth'))
    print(f"Models saved successfully in {checkpoint_dir}!")


    # --- Plotting Results ---
    plot_rewards(episode_rewards_prey, episode_rewards_adversaries)

    # --- Save Training Results and Configuration ---
    results = {
        "prey_rewards": episode_rewards_prey,
        "adversary_rewards": [[float(r) for r in adv_rewards] for adv_rewards in episode_rewards_adversaries],
        "config": {
            "NUM_EPISODES": config.NUM_EPISODES,
            "MAX_STEPS_PER_EPISODE": config.MAX_STEPS_PER_EPISODE,
            "REPLAY_BUFFER_CAPACITY": config.REPLAY_BUFFER_CAPACITY,
            "BATCH_SIZE": config.BATCH_SIZE,
            "LEARNING_RATE": config.LEARNING_RATE,
            "GAMMA": config.GAMMA,
            "TAU": config.TAU,
            "ENV_CONFIG": config.ENV_CONFIG,
            "ALTERNATING_TRAINING": config.ALTERNATING_TRAINING,
            "TRAINING_INTERVAL": config.TRAINING_INTERVAL,
            "INITIAL_TRAINING_AGENT": config.INITIAL_TRAINING_AGENT,
            "PREVIOUS_CHECKPOINT_TIME_STAMP": config.TIME_STAMP if config.RESUME_TRAINING else None
        }
    }
    with open(os.path.join(checkpoint_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Training results and configuration saved successfully in {checkpoint_dir}!")

if __name__ == "__main__":
    train()