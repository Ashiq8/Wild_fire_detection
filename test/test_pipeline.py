import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from src.model_unetpp import UNetPlusPlus
from src.explainability.gradcam import SegmentationGradCAM
from src.vlm.vlm_explainer import VLMExplainer


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "data/images/image_0.jpg"

CHECKPOINT_PATH = (
    "outputs/checkpoints/best_unetpp_aug.pth"
)

OUTPUT_DIR = Path("outputs/explainability")

VLM_OUTPUT_DIR = Path("outputs/vlm")


# ============================================================
# Helper functions
# ============================================================

def prepare_image(image_path):
    """
    Load and prepare an image using the same basic
    preprocessing used during training.
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize((256, 256))

    image_array = np.array(image).astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1)
    ).unsqueeze(0)

    return image, tensor


def create_prediction_overlay(image, prediction):
    """
    Create an image where predicted fire pixels
    are highlighted in red.
    """

    original = np.array(image).copy()

    mask = prediction.astype(bool)

    fire_overlay = np.zeros_like(original)

    fire_overlay[mask] = [255, 0, 0]

    blended = cv2.addWeighted(
        original,
        0.75,
        fire_overlay,
        0.25,
        0
    )

    return blended


def create_gradcam_overlay(image, cam):
    """
    Create Grad-CAM heatmap and overlay.
    """

    original = np.array(image).copy()

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap,
        0.40,
        0
    )

    return heatmap, overlay


def create_explainability_panel(
    original,
    prediction,
    heatmap,
    gradcam_overlay
):
    """
    Create a 2x2 explainability visualization.

    1. Original
    2. Segmentation
    3. Grad-CAM
    4. Grad-CAM overlay
    """

    original = np.array(original)

    # -----------------------------
    # Prediction visualization
    # -----------------------------

    prediction_rgb = np.zeros_like(original)

    prediction_rgb[prediction.astype(bool)] = [
        255,
        0,
        0
    ]

    prediction_overlay = cv2.addWeighted(
        original,
        0.70,
        prediction_rgb,
        0.30,
        0
    )

    # -----------------------------
    # Add labels
    # -----------------------------

    images = [
        ("Original", original),
        ("U-Net++ Prediction", prediction_overlay),
        ("Grad-CAM", heatmap),
        ("Grad-CAM Overlay", gradcam_overlay),
    ]

    panels = []

    for title, img in images:

        panel = img.copy()

        cv2.rectangle(
            panel,
            (0, 0),
            (256, 30),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            panel,
            title,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        panels.append(panel)

    # -----------------------------
    # 2 x 2 panel
    # -----------------------------

    top = np.hstack(
        [panels[0], panels[1]]
    )

    bottom = np.hstack(
        [panels[2], panels[3]]
    )

    final_panel = np.vstack(
        [top, bottom]
    )

    return final_panel


def clean_vlm_output(text):
    """
    Remove the conversation prefix that can appear
    in SmolVLM's decoded output.
    """

    if "Assistant:" in text:

        text = text.split(
            "Assistant:",
            1
        )[1]

    return text.strip()


# ============================================================
# Main pipeline
# ============================================================

def main():

    print("=" * 70)
    print("WILDFIRE SEGMENTATION + GRAD-CAM + VLM PIPELINE")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VLM_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load U-Net++
    # --------------------------------------------------------

    print("\n[1/6] Loading U-Net++...")

    model = UNetPlusPlus(
        num_classes=1
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("U-Net++ loaded.")

    # --------------------------------------------------------
    # 2. Load image
    # --------------------------------------------------------

    print("\n[2/6] Loading wildfire image...")

    image, image_tensor = prepare_image(
        IMAGE_PATH
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    print("Image:", IMAGE_PATH)

    # --------------------------------------------------------
    # 3. Generate segmentation
    # --------------------------------------------------------

    print("\n[3/6] Generating fire segmentation...")

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probability = torch.sigmoid(
            logits
        )

        prediction = (
            probability > 0.5
        )

    prediction_np = (
        prediction[0, 0]
        .cpu()
        .numpy()
    )

    probability_np = (
        probability[0, 0]
        .cpu()
        .numpy()
    )

    fire_pixels = int(
        prediction_np.sum()
    )

    total_pixels = prediction_np.size

    fire_percentage = (
        fire_pixels /
        total_pixels
    ) * 100

    print(
        f"Predicted fire area: "
        f"{fire_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # 4. Grad-CAM
    # --------------------------------------------------------

    print("\n[4/6] Generating Grad-CAM...")

    target_layer = (
        model.model.decoder
        .blocks["x_0_4"]
        .conv2
    )

    gradcam = SegmentationGradCAM(
        model=model,
        target_layer=target_layer
    )

    cam, _ = gradcam.generate(
        image_tensor
    )

    cam = (
        cam[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "Grad-CAM shape:",
        cam.shape
    )

    # Remove hooks after Grad-CAM
    gradcam.remove_hooks()

    # --------------------------------------------------------
    # 5. Create explainability visualization
    # --------------------------------------------------------

    print(
        "\n[5/6] Creating explainability visualization..."
    )

    heatmap, gradcam_overlay = (
        create_gradcam_overlay(
            image,
            cam
        )
    )

    prediction_overlay = (
        create_prediction_overlay(
            image,
            prediction_np
        )
    )

    panel = create_explainability_panel(
        original=image,
        prediction=prediction_np,
        heatmap=heatmap,
        gradcam_overlay=gradcam_overlay
    )

    # Save individual outputs
    cv2.imwrite(
        str(
            OUTPUT_DIR /
            "prediction_overlay.png"
        ),
        cv2.cvtColor(
            prediction_overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR /
            "gradcam_heatmap.png"
        ),
        cv2.cvtColor(
            heatmap,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR /
            "gradcam_overlay.png"
        ),
        cv2.cvtColor(
            gradcam_overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    # Save final 2x2 panel
    cv2.imwrite(
        str(
            OUTPUT_DIR /
            "explainability_panel.png"
        ),
        cv2.cvtColor(
            panel,
            cv2.COLOR_RGB2BGR
        )
    )

    print(
        "Saved explainability outputs to:",
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # 6. VLM
    # --------------------------------------------------------

    print("\n[6/6] Loading SmolVLM...")

    vlm = VLMExplainer(
        device=DEVICE
    )

    # Give VLM the combined explainability image
    vlm_image = Image.fromarray(
        panel
    )

    print(
        "Generating VLM explanation..."
    )

    explanation = vlm.explain(
        image=vlm_image
    )

    explanation = clean_vlm_output(
        explanation
    )

    # Save explanation
    explanation_path = (
        VLM_OUTPUT_DIR /
        "explanation.txt"
    )

    with open(
        explanation_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(explanation)

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL VLM EXPLANATION")
    print("=" * 70)

    print(explanation)

    print("=" * 70)

    print(
        "\nExplainability panel:"
    )

    print(
        OUTPUT_DIR /
        "explainability_panel.png"
    )

    print(
        "\nVLM explanation:"
    )

    print(
        explanation_path
    )


if __name__ == "__main__":
    main()