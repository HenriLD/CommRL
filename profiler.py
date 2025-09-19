import cProfile
import pstats
from train import train

if __name__ == "__main__":
    # Create a Profile object
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Run the function you want to profile
    train()

    # Stop profiling
    profiler.disable()

    # Create a stats object from the profiler
    stats = pstats.Stats(profiler).sort_stats('cumulative')

    # Optionally, save the stats to a file for later analysis
    stats.dump_stats('training_profile.prof')