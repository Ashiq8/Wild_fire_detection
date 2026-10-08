import torch

from src.module.aspp import ASPP


def main():

    x = torch.randn(2, 512, 8, 8)

    model = ASPP(
        in_channels=512,
        out_channels=512,
    )

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()