import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention gate for refining an encoder skip feature
    using a decoder gating feature.

    x: encoder skip feature
    g: decoder feature / gating signal
    """

    def __init__(self, skip_channels, gate_channels, inter_channels=None):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        self.theta_x = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(inter_channels)
        )

        self.phi_g = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1,
                bias=True
            ),
            nn.BatchNorm2d(1)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        """
        x = encoder skip feature
        g = decoder gating feature
        """

        # Resize decoder feature to skip spatial size
        g = nn.functional.interpolate(
            g,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)

        attention = self.relu(theta_x + phi_g)

        attention = self.psi(attention)
        attention = self.sigmoid(attention)

        return x * attention