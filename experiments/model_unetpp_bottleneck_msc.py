import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from experiments.modules.multiscale_context import MultiScaleContext


class UNetPlusPlusBottleneckMSC(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )

        # ResNet34 produces 512 channels at the bottleneck.
        self.multiscale = MultiScaleContext(
            in_channels=512,
            out_channels=512
        )

    def forward(self, x):

        # 1. Extract encoder features
        features = self.model.encoder(x)

        # 2. Deepest encoder feature = bottleneck
        bottleneck = features[-1]

        # 3. Multi-scale context at bottleneck
        bottleneck = self.multiscale(bottleneck)

        # 4. Replace original bottleneck
        features = list(features)
        features[-1] = bottleneck

        # 5. U-Net++ decoder
        decoder_output = self.model.decoder(features)

        # 6. Segmentation head
        mask = self.model.segmentation_head(decoder_output)

        return mask