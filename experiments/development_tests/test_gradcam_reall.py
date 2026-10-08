import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model_unetpp import UNetPlusPlus
from src.explainability.gradcam import SegmentationGradCAM


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # 1. Load trained U-Net++
    # --------------------------------------------------

    model = UNetPlusPlus(num_classes=1).to(device)

    checkpoint_path = "outputs/checkpoints/best_unetpp_aug.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    print("Checkpoint loaded:", checkpoint_path)

    # --------------------------------------------------
    # 2. Load one real FLAME image
    # --------------------------------------------------

    image_path = "data/images/image_0.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Same size used during training
    image_resized = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_LINEAR
    )

    # Keep copy for visualization
    display_image = image_resized.copy()

    # Same normalization as dataset.py
    image_tensor = image_resized.astype(np.float32) / 255.0

    image_tensor = np.transpose(
        image_tensor,
        (2, 0, 1)
    )

    image_tensor = torch.from_numpy(
        image_tensor
    ).unsqueeze(0).to(device)

    # --------------------------------------------------
    # 3. Set Grad-CAM target layer
    # --------------------------------------------------

    target_layer = (
        model.model.decoder
        .blocks["x_0_4"]
        .conv2
    )

    gradcam = SegmentationGradCAM(
        model=model,
        target_layer=target_layer
    )

    # --------------------------------------------------
    # 4. Generate Grad-CAM + segmentation
    # --------------------------------------------------

    cam, logits = gradcam.generate(image_tensor)

    probability = torch.sigmoid(logits)

    prediction = (
        probability > 0.5
    ).float()

    # Convert tensors → numpy
    cam = cam[0, 0].detach().cpu().numpy()

    prediction = (
        prediction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------
    # 5. Create heatmap overlay
    # --------------------------------------------------

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        display_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    # --------------------------------------------------
    # 6. Visualize
    # --------------------------------------------------

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(display_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(prediction, cmap="gray")
    plt.title("U-Net++ Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    gradcam.remove_hooks()


if __name__ == "__main__":
    main()