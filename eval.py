import torch
import time
from pettingzoo.mpe import simple_tag_v3

# Import project components
import config
from sac_agent import SACAgent

def evaluate():
    """Function to evaluate the trained policies."""
    print(f"Using device: {config.DEVICE}")

    # --- Environment Setup ---
    env = simple_tag_v3.parallel_env(render_mode="human", **config.ENV_CONFIG)
    
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
    # Load trained model weights
    adversary_agent.actor.load_state_dict(torch.load("models/sac_simple_tag/adversary_actor.pth"))
    adversary_agent.critic.load_state_dict(torch.load("models/sac_simple_tag/adversary_critic.pth"))


    # Prey Agent
    prey_obs_space = env.observation_space(prey_ids[0])
    prey_action_space = env.action_space(prey_ids[0])
    prey_agent = SACAgent(
        state_dim=prey_obs_space.shape[0],
        action_dim=prey_action_space.shape[0],
        max_action=1.0,
        device=config.DEVICE
    )
    # Load trained model weights
    prey_agent.actor.load_state_dict(torch.load("models/sac_simple_tag/prey_actor.pth"))
    prey_agent.critic.load_state_dict(torch.load("models/sac_simple_tag/prey_critic.pth"))


    # --- Evaluation Loop ---
    obs, _ = env.reset()
    for _ in range(config.MAX_STEPS_PER_EPISODE):
        env.render()
        
        env_actions = {}
        # --- Action Selection and Rescaling ---
        for agent_id in adversary_ids:
            if agent_id in obs:
                act = adversary_agent.select_action(obs[agent_id])
                env_actions[agent_id] = (act + 1.0) / 2.0
        
        for agent_id in prey_ids:
             if agent_id in obs:
                act = prey_agent.select_action(obs[agent_id])
                env_actions[agent_id] = (act + 1.0) / 2.0

        next_obs, _, terminations, truncations, _ = env.step(env_actions)
        obs = next_obs

        if any(terminations.values()) or any(truncations.values()):
            break

        # Add a small delay to make rendering smoother
        time.sleep(0.05)

    env.close()

if __name__ == "__main__":
    evaluate()