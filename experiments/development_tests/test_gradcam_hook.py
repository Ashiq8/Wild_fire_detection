import torch

from src.model_unetpp import UNetPlusPlus


def main():

    model = UNetPlusPlus(num_classes=1)
    model.eval()

    target_layer = model.model.decoder.blocks["x_0_4"].conv2

    activations = {}

    def hook(module, inputs, output):
        activations["feature"] = output

    handle = target_layer.register_forward_hook(hook)

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        output = model(x)

    handle.remove()

    print("Model output :", output.shape)
    print("Target feature:", activations["feature"].shape)


if __name__ == "__main__":
    main()