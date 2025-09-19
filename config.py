# config.py
import torch

# --- Device Configuration ---
# Check for DirectML availability and set it as the device
try:
    import torch_directml
    # torch_directml.device() will select the best available DML device
    DEVICE = torch_directml.device()
    print(f"Using DirectML device: {torch_directml.device_name(0)}")
except (ImportError, ModuleNotFoundError):
    # Fallback to CUDA if DirectML is not available
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    # Fallback to CPU if no GPU is available
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU device")

 
# --- Training Hyperparameters ---
NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 100
REPLAY_BUFFER_CAPACITY = 256_000
BATCH_SIZE = 16384
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
TAU = 0.005   # Soft update factor
RESUME_TRAINING = False # Whether to load from checkpoints if available
TIME_STAMP = None

# --- Alternating Training Configuration ---
ALTERNATING_TRAINING = True
TRAINING_INTERVAL = 1000  # Number of episodes before switching agent group
INITIAL_TRAINING_AGENT = "adversary" # Can be "adversary" or "prey"

# --- Environment Configuration (This is the missing part) ---
ENV_CONFIG = {
    "num_good": 4,
    "num_adversaries": 2,
    "num_obstacles": 0,
    "max_cycles": MAX_STEPS_PER_EPISODE,
    "continuous_actions": True,
}