import torch.nn as nn
import segmentation_models_pytorch as smp

from src.module.aspp import ASPP


class UNetPlusPlusASPP(nn.Module):

    def __init__(self, num_classes=1):

        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        # ASPP operates on the 512-channel bottleneck
        self.aspp = ASPP(
            in_channels=512,
            out_channels=512,
        )

    def forward(self, x):

        # Extract encoder features
        features = self.model.encoder(x)

        # Get deepest feature
        bottleneck = features[-1]

        # Apply ASPP
        bottleneck = self.aspp(bottleneck)

        # Replace original bottleneck
        features = list(features)
        features[-1] = bottleneck

        # Decode
        decoder_output = self.model.decoder(features)

        # Segmentation
        output = self.model.segmentation_head(
            decoder_output
        )

        return output