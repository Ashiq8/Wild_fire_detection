import torch


def dice_score(predictions, targets, threshold=0.5, smooth=1.0):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (
        (2.0 * intersection + smooth)
        / (predictions.sum() + targets.sum() + smooth)
    )

    return dice.item()


def iou_score(predictions, targets, threshold=0.5, smooth=1.0):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()