import torch

from src.config import DEVICE
from src.model_segformer import UNet


# Create model
model = UNet(num_classes=1)

# Move model to device
model = model.to(DEVICE)

# Create dummy input
x = torch.randn(1, 3, 256, 256).to(DEVICE)

# Forward pass
with torch.no_grad():
    output = model(x)

# Count trainable parameters
total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

# Print results
print("Device:", DEVICE)
print("Input shape :", x.shape)
print("Output shape:", output.shape)
print("Trainable parameters:", total_params)