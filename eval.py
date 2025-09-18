import torch
import time
from pettingzoo.mpe import simple_tag_v3

# Import project components
import config
from sac_agent import SACAgent
import pygame
from pragmatic_wrapper import PragmaticWrapper
from utils import load_checkpoint

clock = pygame.time.Clock()

CHECKPOINT_TIMESTAMP = "20250918_130405"
NUM_EPISODES = 100

def evaluate():
    """Function to evaluate the trained policies."""
    print(f"Using device: {config.DEVICE}")

    # --- Environment Setup ---
    # The render_mode="human" argument initializes the pygame window
    env = simple_tag_v3.parallel_env(render_mode="human", **config.ENV_CONFIG)
    env = PragmaticWrapper(env)
    adversary_ids = [agent for agent in env.possible_agents if 'adversary' in agent]
    prey_ids = [agent for agent in env.possible_agents if 'agent' in agent]

    # --- Agent Initialization ---
    # Adversary Agent
    adv_obs_space = env.observation_space(adversary_ids[0])
    adv_action_space = env.action_space(adversary_ids[0])
    adversary_agent = SACAgent(
        state_dim=adv_obs_space.shape[0],
        action_dim=adv_action_space.shape[0],
        max_action=1.0,
        device=config.DEVICE
    )

    # Prey Agent
    prey_obs_space = env.observation_space(prey_ids[0])
    prey_action_space = env.action_space(prey_ids[0])
    prey_agent = SACAgent(
        state_dim=prey_obs_space.shape[0],
        action_dim=prey_action_space.shape[0],
        max_action=1.0,
        device=config.DEVICE
    )

    # --- Load Trained Models ---
    load_checkpoint(adversary_agent, prey_agent, timestamp=CHECKPOINT_TIMESTAMP, safe_mode=True)

    # --- Evaluation Loop ---
    obs, _ = env.reset()
    
    # Render the initial state before the loop starts
    env.render()
    time.sleep(1) # Pause for a second to see the start

    for episode in range(NUM_EPISODES):
        env_actions = {}
        # --- Action Selection ---
        for agent_id in adversary_ids:
            if agent_id in obs:
                with torch.no_grad():
                    act = adversary_agent.select_action(obs[agent_id], evaluate=True)
                    env_actions[agent_id] = (act + 1.0) / 2.0
        
        for agent_id in prey_ids:
             if agent_id in obs:
                with torch.no_grad():
                    act = prey_agent.select_action(obs[agent_id], evaluate=True)
                    env_actions[agent_id] = (act + 1.0) / 2.0

        # --- Step the Environment ---
        next_obs, _, terminations, truncations, _ = env.step(env_actions)
        obs = next_obs

        # --- Render the NEW state ---
        # This call also handles pygame events to prevent freezing.
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        if any(terminations.values()) or any(truncations.values()):
            break

        # This delay controls the speed of the visualization.
        # If it's still freezing, try a smaller value like 0.01
        print(f"Step {episode+1}/{config.MAX_STEPS_PER_EPISODE} completed.")

        clock.tick(5)

    env.close()
    print("Evaluation finished.")

if __name__ == "__main__":
    evaluate()