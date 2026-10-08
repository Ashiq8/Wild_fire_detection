import torch

from src.losses import BCEDiceLoss


predictions = torch.randn(8, 1, 256, 256)
targets = torch.randint(0, 2, (8, 1, 256, 256)).float()

criterion = BCEDiceLoss()

loss = criterion(predictions, targets)

print("Loss:", loss.item())