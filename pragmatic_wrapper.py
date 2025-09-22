import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
import pygame
from gymnasium import spaces # Import the spaces module

class PragmaticWrapper(BaseParallelWrapper):
    """
    A refined wrapper that adds a pragmatic reward bonus based on relative
    distances already present in the agent's observation space.

    This wrapper assigns a "meaning" to each adversary agent, which is a
    preference for a specific prey. A reward bonus is given to the adversary
    if it moves closer to its preferred prey.
    """
    def __init__(self, env, pragmatic_reward_bonus=0.01):
        """
        Initializes the wrapper.

        Args:
            env: The environment to wrap.
            pragmatic_reward_bonus (float): The reward bonus for moving
                                             closer to the preferred prey.
        """
        super().__init__(env)

        # --- Define these attributes at the beginning ---
        self.adversary_ids = [agent for agent in self.possible_agents if 'adversary' in agent]
        self.prey_ids = [agent for agent in self.possible_agents if 'agent' in agent]
        self.num_prey = len(self.prey_ids)

        self.pragmatic_reward_bonus = pragmatic_reward_bonus

        self.render_mode = self.env.unwrapped.render_mode
        if self.render_mode == "human":
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)

        self.env.unwrapped.render_mode = None
        
        # This mapping is crucial for finding the correct slice in the observation vector
        self._agent_id_to_idx = {agent_id: i for i, agent_id in enumerate(self.possible_agents)}
        self._prey_id_to_idx = {agent_id: i for i, agent_id in enumerate(self.prey_ids)}


        self.adversary_meanings = {}
        self._assign_meanings()
        self.last_obs = None
        self.agent_rewards = {agent_id: 0 for agent_id in self.possible_agents}
        
        # --- MODIFICATION: Create a new dict for our spaces ---
        self._my_observation_spaces = {
            agent: super().observation_space(agent) for agent in self.possible_agents
        }
        
        # Update the spaces for adversaries in our new dict
        for agent_id in self.adversary_ids:
            original_obs_space = self._my_observation_spaces[agent_id]
            low = np.concatenate([
                original_obs_space.low, 
                np.zeros(self.num_prey, dtype=np.float32), 
                np.zeros(self.num_prey, dtype=np.float32)
            ])
            high = np.concatenate([
                original_obs_space.high, 
                np.ones(self.num_prey, dtype=np.float32), 
                np.ones(self.num_prey, dtype=np.float32)
            ])
            self._my_observation_spaces[agent_id] = spaces.Box(low=low, high=high, dtype=np.float32)

    # --- CORRECTION: Explicitly override the observation_space method ---
    def observation_space(self, agent):
        """
        Returns the observation space for a specific agent from our custom dictionary.
        """
        return self._my_observation_spaces[agent]


    def _get_wrapped_obs(self, obs):
        """
        Wraps the original observations with the preferred target and belief vectors for adversaries.
        """
        wrapped_obs = {}
        for agent_id, original_obs in obs.items():
            if agent_id in self.adversary_ids:
                # Create the one-hot encoded vector for the preferred prey
                preferred_prey_id = self.adversary_meanings[agent_id]
                prey_idx = self._prey_id_to_idx[preferred_prey_id]
                preferred_target_vec = np.zeros(self.num_prey)
                preferred_target_vec[prey_idx] = 1.0

                # Create the belief vector (all zeros for now)
                belief_vec = np.zeros(self.num_prey)

                wrapped_obs[agent_id] = np.concatenate([original_obs, preferred_target_vec, belief_vec])
            else:
                # Prey observations are not changed
                wrapped_obs[agent_id] = original_obs

        return wrapped_obs

    def reset(self, **kwargs):
        """
        Resets the environment and re-assigns meanings to the adversaries.
        """
        obs, info = self.env.reset(**kwargs)
        self._assign_meanings()
        self.agent_rewards = {agent_id: 0 for agent_id in self.possible_agents}
        
        wrapped_obs = self._get_wrapped_obs(obs)
        self.last_obs = wrapped_obs
        return wrapped_obs, info

    def step(self, actions):
        """
        Steps the environment and adds pragmatic rewards based on changes
        in relative distance to the preferred prey.
        """
        if self.last_obs is None:
            # This can happen if step() is called before reset()
            self.last_obs, _ = self.reset()

        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # First, add the pragmatic bonus to the rewards dictionary
        for adv_id in self.adversary_ids:
            if adv_id in self.last_obs and adv_id in next_obs:
                # --- CORRECTION: Use super() to get original observation space size for slicing ---
                original_obs_len = super().observation_space(adv_id).shape[0]
                original_last_obs = self.last_obs[adv_id][:original_obs_len]
                
                original_next_obs = next_obs[adv_id]

                preferred_prey_id = self.adversary_meanings[adv_id]
                
                vec_to_prey_before = self._get_relative_pos(original_last_obs, adv_id, preferred_prey_id)
                vec_to_prey_after = self._get_relative_pos(original_next_obs, adv_id, preferred_prey_id)

                if vec_to_prey_before is not None and vec_to_prey_after is not None:
                    dist_before = np.linalg.norm(vec_to_prey_before)
                    dist_after = np.linalg.norm(vec_to_prey_after)
                    
                    if dist_after < dist_before:
                        rewards[adv_id] += self.pragmatic_reward_bonus * (dist_before - dist_after)
        
        # Now, update the cumulative rewards for rendering
        for agent_id, reward in rewards.items():
            self.agent_rewards[agent_id] += reward

        # Finally, wrap the next observation and return
        wrapped_next_obs = self._get_wrapped_obs(next_obs)
        self.last_obs = wrapped_next_obs
        return wrapped_next_obs, rewards, terminations, truncations, infos

    def _get_relative_pos(self, obs_vector, observer_id, target_id):
        """
        Extracts the relative position of a target agent from the
        observer agent's observation vector.
        """
        other_pos_start_idx = 8 
        other_agent_ids = [id for id in self.possible_agents if id != observer_id]
        
        try:
            list_idx = other_agent_ids.index(target_id)
        except ValueError:
            return None

        start = other_pos_start_idx + list_idx * 2
        end = start + 2
        
        return obs_vector[start:end]

    def _assign_meanings(self):
        """
        Assigns a preferred prey to each adversary. This represents the
        adversary's communicative "meaning" for the episode.
        """
        for adv_id in self.adversary_ids:
            self.adversary_meanings[adv_id] = np.random.choice(self.prey_ids)

    def render(self):
        """
        Renders the environment from scratch, including agents, landmarks,
        and custom lines. This method is the single source of truth for rendering.
        """
        if self.render_mode != "human":
            return

        try:
            unwrapped_env = self.env.unwrapped

            if not unwrapped_env.renderOn:
                unwrapped_env.enable_render(self.render_mode)
            
            screen = unwrapped_env.screen
            screen.fill((255, 255, 255))

            all_poses = [entity.state.p_pos for entity in unwrapped_env.world.entities]
            cam_range = np.max(np.abs(np.array(all_poses)))
            if cam_range == 0: cam_range = 1

            for entity in unwrapped_env.world.entities:
                x, y = entity.state.p_pos
                y *= -1
                scr_x = (x / cam_range) * 315 + 350
                scr_y = (y / cam_range) * 315 + 350
                radius = entity.size * 350
                pygame.draw.circle(screen, entity.color * 200, (scr_x, scr_y), radius)
                pygame.draw.circle(screen, (0, 0, 0), (scr_x, scr_y), radius, 1)

            agent_map = {agent.name: agent for agent in unwrapped_env.world.agents}
            line_color = (255, 0, 0)
            line_width = 2
            
            for adv_id, prey_id in self.adversary_meanings.items():
                if adv_id in agent_map and prey_id in agent_map:
                    adv_agent = agent_map[adv_id]
                    prey_agent = agent_map[prey_id]
                    start_x, start_y = adv_agent.state.p_pos
                    start_y *= -1
                    start_pos = ((start_x / cam_range) * 315 + 350, (start_y / cam_range) * 315 + 350)
                    end_x, end_y = prey_agent.state.p_pos
                    end_y *= -1
                    end_pos = ((end_x / cam_range) * 315 + 350, (end_y / cam_range) * 315 + 350)
                    pygame.draw.line(screen, line_color, start_pos, end_pos, line_width)
            
            y_offset = 10
            for agent_id, reward in self.agent_rewards.items():
                text = self.font.render(f"{agent_id}: {reward:.2f}", True, (0, 0, 0))
                screen.blit(text, (10, y_offset))
                y_offset += 20

            pygame.display.flip()

        except AttributeError as e:
            print(f"Rendering failed: {e}")
            pass

    def close(self):
        self.env.close()