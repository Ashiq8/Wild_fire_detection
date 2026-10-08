import cv2
import torch
import numpy as np
from PIL import Image

from src.model_unetpp import UNetPlusPlus
from src.vlm.vlm_explainer import VLMExplainer


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------
    # 1. Load U-Net++
    # -----------------------------------------

    print("Loading U-Net++...")

    model = UNetPlusPlus(num_classes=1).to(device)

    checkpoint = torch.load(
        "outputs/checkpoints/best_unetpp_aug.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    print("U-Net++ loaded.")

    # -----------------------------------------
    # 2. Load real FLAME image
    # -----------------------------------------

    image_path = "data/images/image_0.jpg"

    image = Image.open(image_path).convert("RGB")

    image_resized = image.resize((256, 256))

    image_array = np.array(image_resized).astype(
        np.float32
    ) / 255.0

    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # -----------------------------------------
    # 3. Generate segmentation
    # -----------------------------------------

    print("Generating segmentation...")

    with torch.no_grad():

        logits = model(image_tensor)

        probability = torch.sigmoid(logits)

        prediction = (
            probability > 0.5
        ).float()

    # -----------------------------------------
    # 4. Create fire-highlighted image
    # -----------------------------------------

    mask = prediction[0, 0].cpu().numpy()

    mask = (mask * 255).astype(np.uint8)

    original = np.array(image_resized).copy()

    # Copy original image
    overlay = original.copy()

    # Fire pixels
    fire_pixels = mask > 0

    # Highlight predicted fire in red
    overlay[fire_pixels] = [255, 0, 0]

    # Blend original + highlighted mask
    blended = cv2.addWeighted(
        original,
        0.65,
        overlay,
        0.35,
        0
    )

    overlay_image = Image.fromarray(blended)

    # -----------------------------------------
    # 5. Load VLM
    # -----------------------------------------

    vlm = VLMExplainer(device=device)

    # -----------------------------------------
    # 6. Ask VLM to explain
    # -----------------------------------------

    print("Generating VLM explanation...")

    explanation = vlm.explain(
        image=overlay_image
    )

    # -----------------------------------------
    # 7. Print result
    # -----------------------------------------

    print("\n")
    print("=" * 60)
    print("VLM EXPLANATION")
    print("=" * 60)
    print(explanation)
    print("=" * 60)


if __name__ == "__main__":
    main()