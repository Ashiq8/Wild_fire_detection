import torch
import torch.nn as nn


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super().__init__()

        branch_channels = out_channels // 4

        # 1 × 1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # 3 × 3 convolution, dilation = 1
        self.branch2 = ASPPConv(
            in_channels,
            branch_channels,
            dilation=1,
        )

        # 3 × 3 convolution, dilation = 6
        self.branch3 = ASPPConv(
            in_channels,
            branch_channels,
            dilation=6,
        )

        # 3 × 3 convolution, dilation = 12
        self.branch4 = ASPPConv(
            in_channels,
            branch_channels,
            dilation=12,
        )

        # Fuse all branches
        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Combine multi-scale features
        x = torch.cat(
            [b1, b2, b3, b4],
            dim=1,
        )

        # Fuse them
        return self.project(x)