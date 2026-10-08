import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from experiments.modules.multiscale_context import MultiScaleContext


class UNetPlusPlusMultiScale(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )

        # U-Net++ decoder produces 16 channels.
        self.multiscale = MultiScaleContext(
            in_channels=16,
            out_channels=16
        )

    def forward(self, x):

        # Encoder
        features = self.model.encoder(x)

        # U-Net++ decoder
        decoder_output = self.model.decoder(features)

        # Multi-scale context refinement
        decoder_output = self.multiscale(decoder_output)

        # Final segmentation mask
        mask = self.model.segmentation_head(decoder_output)

        return mask