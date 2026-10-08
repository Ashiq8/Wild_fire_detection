import torch

from experiments.modules.multiscale_context import MultiScaleContext


def test_multiscale_context():
    x = torch.randn(2, 16, 64, 64)

    model = MultiScaleContext(
        in_channels=16,
        out_channels=16,
    )

    y = model(x)

    assert y.shape == x.shape

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Multi-Scale Context test passed.")


if __name__ == "__main__":
    test_multiscale_context()