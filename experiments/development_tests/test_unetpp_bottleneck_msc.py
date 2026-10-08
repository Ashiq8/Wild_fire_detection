import torch

from src.model_unetpp_bottleneck_msc import UNetPlusPlusBottleneckMSC


def main():

    model = UNetPlusPlusBottleneckMSC(num_classes=1)
    model.eval()

    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        features = model.model.encoder(x)

        print("Original bottleneck:", features[-1].shape)

        output = model(x)

    print("Final output:", output.shape)

    assert features[-1].shape == (2, 512, 8, 8)
    assert output.shape == (2, 1, 256, 256)

    print("Bottleneck MSC test passed.")


if __name__ == "__main__":
    main()