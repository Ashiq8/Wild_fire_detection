import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import DEVICE
from experiments.unet_model import UNet
from src.model_unetpp import UNetPlusPlus
from src.dataset import WildfireSegmentationDataset
from experiments.model_segformer import SegFormer
from experiments.model_deeplab import DeepLabV3Plus

from experiments.attention_unet_model import AttentionUNet
from experiments.multiscale_unet_model import MultiScaleUNet


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(DEVICE)

IMAGE_DIR = Path("data/images")
MASK_DIR = Path("data/masks")

OUTPUT_DIR = Path("outputs/final_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "U-Net": "outputs/checkpoints/best_unet_aug.pth",
    "U-Net++": "outputs/checkpoints/best_unetpp_aug.pth",
    "Attention U-Net": "outputs/checkpoints/best_attention_unet.pth",
    "Multi-Scale U-Net": "outputs/checkpoints/best_multiscale_unet.pth",
    "SegFormer": "outputs/checkpoints/best_segformer_aug.pth",
    "DeepLabV3+": "outputs/checkpoints/best_deeplab_aug.pth",
}


# ============================================================
# Model Factory
# ============================================================

def create_models():

    models = {

        "U-Net": UNet(
            num_classes=1
        ),

        "U-Net++": UNetPlusPlus(
            num_classes=1
        ),

        "Attention U-Net": AttentionUNet(
            num_classes=1
        ),

        "Multi-Scale U-Net": MultiScaleUNet(
            num_classes=1
        ),

        "SegFormer": SegFormer(
            num_classes=1
        ),

        "DeepLabV3+": DeepLabV3Plus(
            num_classes=1
        ),
    }

    return models


# ============================================================
# Load Checkpoints
# ============================================================

def load_models():

    models = create_models()

    for name, model in models.items():

        checkpoint_path = CHECKPOINTS[name]

        print(f"Loading {name}...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()

        print(f"  Loaded: {checkpoint_path}")

    return models


# ============================================================
# Prediction
# ============================================================

def predict(model, image):

    image_tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float()

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(image_tensor)

        probability = torch.sigmoid(logits)

        prediction = (
            probability > 0.5
        ).float()

    prediction = prediction.squeeze().cpu().numpy()

    return prediction


# ============================================================
# Convert mask to display
# ============================================================

def prepare_mask(mask):

    if torch.is_tensor(mask):
        mask = mask.squeeze().numpy()

    return (mask > 0.5).astype(np.uint8)


# ============================================================
# Generate qualitative panel
# ============================================================

def generate_panel(
    image,
    ground_truth,
    predictions,
    sample_name
):

    model_names = list(predictions.keys())

    # Input + Ground Truth + 6 models
    total_columns = 2 + len(model_names)

    fig, axes = plt.subplots(
        1,
        total_columns,
        figsize=(24, 4.5)
    )

    # --------------------------------------------------------
    # Input image
    # --------------------------------------------------------

    axes[0].imshow(image)
    axes[0].set_title(
        "Input Image",
        fontsize=12,
        fontweight="bold"
    )
    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1].imshow(
        ground_truth,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth",
        fontsize=12,
        fontweight="bold"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    for i, model_name in enumerate(model_names):

        axes[i + 2].imshow(
            predictions[model_name],
            cmap="gray"
        )

        axes[i + 2].set_title(
            model_name,
            fontsize=11,
            fontweight="bold"
        )

        axes[i + 2].axis("off")

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        "Qualitative Analysis — Wildfire Segmentation",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"{sample_name}_qualitative_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("======================================")
    print("QUALITATIVE SEGMENTATION COMPARISON")
    print("======================================")

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=256
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading models...")

    models = load_models()

    print("\nAll models loaded.")

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    # Start with one image.
    # Later we can change this to [0, 1, 2, 3]
    # for multiple qualitative examples.

    sample_indices = [0, 50, 100, 150, 200, 250]

    for sample_index in sample_indices:

        print("\n--------------------------------------")
        print(
            f"Processing sample index: {sample_index}"
        )
        print("--------------------------------------")

        image_tensor, mask_tensor = dataset[
            sample_index
        ]

        # Convert image tensor -> HWC RGB
        image = (
            image_tensor
            .numpy()
            .transpose(1, 2, 0)
        )

        image = np.clip(
            image,
            0,
            1
        )

        ground_truth = prepare_mask(
            mask_tensor
        )

        predictions = {}

        # ----------------------------------------------------
        # Predict with every model
        # ----------------------------------------------------

        for model_name, model in models.items():

            print(
                f"Predicting with {model_name}..."
            )

            prediction = predict(
                model,
                image
            )

            predictions[
                model_name
            ] = prediction

        # ----------------------------------------------------
        # Generate panel
        # ----------------------------------------------------

        generate_panel(
            image=image,
            ground_truth=ground_truth,
            predictions=predictions,
            sample_name=f"sample_{sample_index:04d}"
        )

    print("\n======================================")
    print("QUALITATIVE COMPARISON COMPLETE")
    print("======================================")


if __name__ == "__main__":
    main()