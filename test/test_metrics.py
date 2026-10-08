import torch

from src.metrics import dice_score, iou_score


predictions = torch.randn(8, 1, 256, 256)
targets = torch.randint(0, 2, (8, 1, 256, 256)).float()

dice = dice_score(predictions, targets)
iou = iou_score(predictions, targets)

print("Dice:", dice)
print("IoU :", iou)