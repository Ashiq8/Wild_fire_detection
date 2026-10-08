import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        dice = (
            (2.0 * intersection + self.smooth)
            / (
                predictions.sum()
                + targets.sum()
                + self.smooth
            )
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):

        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )