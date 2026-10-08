import torch

from src.model_unetpp import UNetPlusPlus
from src.explainability.gradcam import SegmentationGradCAM


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNetPlusPlus(num_classes=1).to(device)
    model.eval()

    target_layer = model.model.decoder.blocks["x_0_4"].conv2

    gradcam = SegmentationGradCAM(
        model=model,
        target_layer=target_layer
    )

    image = torch.randn(1, 3, 256, 256).to(device)

    cam, logits = gradcam.generate(image)

    print("Input       :", image.shape)
    print("Logits      :", logits.shape)
    print("Grad-CAM    :", cam.shape)
    print("CAM min     :", cam.min().item())
    print("CAM max     :", cam.max().item())

    gradcam.remove_hooks()


if __name__ == "__main__":
    main()