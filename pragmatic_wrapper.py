import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

class PragmaticWrapper(BaseParallelWrapper):
    """
    A refined wrapper that adds a pragmatic reward bonus based on relative
    distances already present in the agent's observation space.

    This wrapper assigns a "meaning" to each adversary agent, which is a
    preference for a specific prey. A reward bonus is given to the adversary
    if it moves closer to its preferred prey.
    """
    def __init__(self, env, pragmatic_reward_bonus=10.0):
        """
        Initializes the wrapper.

        Args:
            env: The environment to wrap.
            pragmatic_reward_bonus (float): The reward bonus for moving
                                             closer to the preferred prey.
        """
        super().__init__(env)
        self.pragmatic_reward_bonus = pragmatic_reward_bonus
        
        # Get agent IDs
        self.adversary_ids = [agent for agent in self.possible_agents if 'adversary' in agent]
        self.prey_ids = [agent for agent in self.possible_agents if 'agent' in agent]
        
        # This mapping is crucial for finding the correct slice in the observation vector
        self._agent_id_to_idx = {agent_id: i for i, agent_id in enumerate(self.possible_agents)}

        self.adversary_meanings = {}
        self._assign_meanings()
        self.last_obs = None

    def reset(self, **kwargs):
        """
        Resets the environment and re-assigns meanings to the adversaries.
        """
        obs, info = self.env.reset(**kwargs)
        self._assign_meanings()
        self.last_obs = obs
        return obs, info

    def step(self, actions):
        """
        Steps the environment and adds pragmatic rewards based on changes
        in relative distance to the preferred prey.
        """
        if self.last_obs is None:
            # This can happen if step() is called before reset()
            self.last_obs, _ = self.reset()

        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        for adv_id in self.adversary_ids:
            if adv_id in self.last_obs and adv_id in next_obs:
                preferred_prey_id = self.adversary_meanings[adv_id]
                
                # Get the relative position vectors from the adversary's POV
                vec_to_prey_before = self._get_relative_pos(self.last_obs[adv_id], adv_id, preferred_prey_id)
                vec_to_prey_after = self._get_relative_pos(next_obs[adv_id], adv_id, preferred_prey_id)

                if vec_to_prey_before is not None and vec_to_prey_after is not None:
                    # Calculate the distance (magnitude of the relative position vector)
                    dist_before = np.linalg.norm(vec_to_prey_before)
                    dist_after = np.linalg.norm(vec_to_prey_after)
                    
                    # Add reward bonus if the adversary got closer
                    if dist_after < dist_before:
                        rewards[adv_id] += self.pragmatic_reward_bonus

        self.last_obs = next_obs
        return next_obs, rewards, terminations, truncations, infos

    def _get_relative_pos(self, obs_vector, observer_id, target_id):
        """
        Extracts the relative position of a target agent from the
        observer agent's observation vector.
        """
        # The 'other_pos' block starts at index 8.
        other_pos_start_idx = 8 

        # Find the index for the target agent in the 'other_pos' list
        observer_idx = self._agent_id_to_idx[observer_id]
        target_idx = self._agent_id_to_idx[target_id]
        
        # The 'other_pos' list is ordered by agent index, skipping the observer itself
        other_agent_ids = [id for id in self.possible_agents if id != observer_id]
        
        try:
            list_idx = other_agent_ids.index(target_id)
        except ValueError:
            return None # Should not happen in this scenario

        # Each position is a 2D vector
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
