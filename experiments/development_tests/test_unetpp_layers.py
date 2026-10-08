from src.model_unetpp import UNetPlusPlus


def main():
    model = UNetPlusPlus(num_classes=1)

    print("\n=== U-Net++ Decoder ===")

    for name, module in model.model.decoder.named_modules():
        if name:
            print(f"{name}: {module.__class__.__name__}")


if __name__ == "__main__":
    main()