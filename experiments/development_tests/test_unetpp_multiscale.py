import torch

from src.model_unetpp_multiscale import UNetPlusPlusMultiScale


def main():

    model = UNetPlusPlusMultiScale(num_classes=1)
    model.eval()

    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    assert y.shape == (2, 1, 256, 256)

    print("U-Net++ + Multi-Scale Context test passed.")


if __name__ == "__main__":
    main()