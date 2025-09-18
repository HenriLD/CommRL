import time
from pettingzoo.mpe import simple_tag_v3
from pragmatic_wrapper import PragmaticWrapper
import pygame

clock = pygame.time.Clock()

def run_random_test():
    """
    Initializes the simple_tag_v2 environment, wraps it with the
    PragmaticWrapper, and runs a short episode with random actions to
    demonstrate the rendering of preferred prey lines.
    """
    # Initialize the PettingZoo environment
    # render_mode='human' is required to see the visualization
    env = simple_tag_v3.parallel_env(num_good=3, num_adversaries=3, num_obstacles=0, render_mode='human', max_cycles=200)

    # Wrap the environment with our custom wrapper
    env = PragmaticWrapper(env)

    # Reset the environment to get initial observations
    observations, infos = env.reset()

    print("Starting random agent test...")
    print("Look for red lines connecting adversaries to their preferred prey.")

    # Main loop to run the simulation
    while env.agents:
        # For each agent, select a random action from its action space
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # Step the environment forward with the chosen actions
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # The wrapper's render() method is called here. It will first call the
        # environment's render method, and then draw the lines on top.
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            
        clock.tick(5)


    # Clean up the environment resources
    env.close()
    print("Test finished.")

if __name__ == "__main__":
    run_random_test()
