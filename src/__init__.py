import torch
from src.model_unetpp import UNetPlusPlus


def main():
    model = UNetPlusPlus(num_classes=1)

    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        features = model.model.encoder(x)

    print("\nEncoder feature maps:")

    for i, feature in enumerate(features):
        print(f"Feature {i}: {feature.shape}")


if __name__ == "__main__":
    main()