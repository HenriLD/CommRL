import torch
import torch_directml

if torch_directml.is_available():
    device = torch_directml.device()
    device_name = torch_directml.device_name(0)
    print(f"DirectML is available! Using device: {device_name}")

    # Create a tensor and move it to the DML device
    try:
        x = torch.randn(3, 3).to(device)
        print("Tensor created and moved to DML device successfully:")
        print(x)
        print(f"Tensor is on device: {x.device}")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("DirectML is not available.")