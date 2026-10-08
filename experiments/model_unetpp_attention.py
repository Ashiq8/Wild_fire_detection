import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from src.module.attention import AttentionGate


class UNetPlusPlusAttention(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        # ResNet34 encoder feature channels
        #
        # features:
        # 0 -> 3
        # 1 -> 64
        # 2 -> 64
        # 3 -> 128
        # 4 -> 256
        # 5 -> 512

        self.attention_1 = AttentionGate(
            skip_channels=64,
            gate_channels=512,
            inter_channels=32
        )

        self.attention_2 = AttentionGate(
            skip_channels=64,
            gate_channels=512,
            inter_channels=32
        )

        self.attention_3 = AttentionGate(
            skip_channels=128,
            gate_channels=512,
            inter_channels=64
        )

        self.attention_4 = AttentionGate(
            skip_channels=256,
            gate_channels=512,
            inter_channels=128
        )

    def forward(self, x):

        # Encoder
        features = self.model.encoder(x)

        # Original bottleneck
        bottleneck = features[-1]

        # Use bottleneck as the gating signal
        g = bottleneck

        # Attention on encoder features
        features = list(features)

        features[1] = self.attention_1(
            features[1],
            g
        )

        features[2] = self.attention_2(
            features[2],
            g
        )

        features[3] = self.attention_3(
            features[3],
            g
        )

        features[4] = self.attention_4(
            features[4],
            g
        )

        # Keep bottleneck unchanged
        features[5] = bottleneck

        # U-Net++ decoder
        decoder_output = self.model.decoder(features)

        # Segmentation head
        output = self.model.segmentation_head(
            decoder_output
        )

        return output