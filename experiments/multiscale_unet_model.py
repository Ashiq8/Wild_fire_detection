import torch
import torch.nn as nn


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Split output channels between the three scales
        ch1 = out_channels // 3
        ch2 = out_channels // 3
        ch3 = out_channels - ch1 - ch2

        # Small receptive field
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                ch1,
                kernel_size=3,
                padding=1,
                dilation=1
            ),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True)
        )

        # Medium receptive field
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                ch2,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True)
        )

        # Large receptive field
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                ch3,
                kernel_size=3,
                padding=3,
                dilation=3
            ),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True)
        )

        # Fuse the three scales
        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1
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

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat(
            [x1, x2, x3],
            dim=1
        )

        return self.fusion(x)


class MultiScaleUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Encoder
        self.enc1 = MultiScaleConv(3, 32)
        self.enc2 = MultiScaleConv(32, 64)
        self.enc3 = MultiScaleConv(64, 128)
        self.enc4 = MultiScaleConv(128, 256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Bottleneck
        self.bottleneck = MultiScaleConv(
            256,
            512
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = MultiScaleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = MultiScaleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = MultiScaleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = MultiScaleConv(
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

        # Decoder
        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.out(d1)