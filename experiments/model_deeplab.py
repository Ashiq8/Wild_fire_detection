import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)