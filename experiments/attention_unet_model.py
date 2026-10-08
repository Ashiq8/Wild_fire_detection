import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(
        self,
        gating_channels,
        skip_channels,
        intermediate_channels
    ):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(
                gating_channels,
                intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                intermediate_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g = gating signal from decoder
        x = skip connection feature from encoder
        """

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(
                g1,
                size=x1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        attention = self.relu(g1 + x1)

        attention = self.psi(attention)

        return x * attention


class AttentionUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionGate(
            gating_channels=256,
            skip_channels=256,
            intermediate_channels=128
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionGate(
            gating_channels=128,
            skip_channels=128,
            intermediate_channels=64
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionGate(
            gating_channels=64,
            skip_channels=64,
            intermediate_channels=32
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionGate(
            gating_channels=32,
            skip_channels=32,
            intermediate_channels=16
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        # Final segmentation layer
        self.out = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder + Attention
        d4 = self.up4(b)

        e4_att = self.att4(
            d4,
            e4
        )

        d4 = torch.cat(
            [d4, e4_att],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        e3_att = self.att3(
            d3,
            e3
        )

        d3 = torch.cat(
            [d3, e3_att],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        e2_att = self.att2(
            d2,
            e2
        )

        d2 = torch.cat(
            [d2, e2_att],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        e1_att = self.att1(
            d1,
            e1
        )

        d1 = torch.cat(
            [d1, e1_att],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.out(d1)