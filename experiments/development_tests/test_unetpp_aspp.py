import torch

from experiments.model_unetpp_aspp import UNetPlusPlusASPP


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNetPlusPlusASPP(
        num_classes=1
    ).to(device)

    x = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():

        y = model(x)

    print("Device      :", device)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()