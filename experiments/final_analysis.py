import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import random_split

from src.config import (
    IMAGE_DIR,
    MASK_DIR,
    IMAGE_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    DEVICE,
)

from src.dataset import WildfireSegmentationDataset
from src.unet_model import UNet
from src.model_segformer import SegFormer
from src.model_deeplab import DeepLabV3Plus
from src.model_unetpp import UNetPlusPlus


# ==========================================================
# SETTINGS
# ==========================================================

CHECKPOINTS = {
    "U-Net": "outputs/checkpoints/best_unet.pth",
    "SegFormer": "outputs/checkpoints/best_segformer.pth",
    "DeepLabV3+": "outputs/checkpoints/best_deeplab.pth",
    "U-Net++": "outputs/checkpoints/best_unetpp.pth",
}

OUTPUT_DIR = "outputs/final_analysis"
QUALITATIVE_DIR = os.path.join(
    OUTPUT_DIR,
    "qualitative"
)
PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

NUM_QUALITATIVE_CASES = 8


# ==========================================================
# CLEAN OLD FINAL ANALYSIS OUTPUT
# ==========================================================

def prepare_output_directory():

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(QUALITATIVE_DIR)
    os.makedirs(PLOT_DIR)


# ==========================================================
# CREATE SAME TEST SPLIT
# ==========================================================

def create_test_dataset(dataset):

    total_size = len(dataset)

    train_size = int(
        total_size * TRAIN_RATIO
    )

    val_size = int(
        total_size * VAL_RATIO
    )

    test_size = (
        total_size
        - train_size
        - val_size
    )

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return test_dataset


# ==========================================================
# LOAD MODELS
# ==========================================================

def load_models():

    models = {}

    print("\nLoading models...")

    # U-Net
    model = UNet(num_classes=1).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINTS["U-Net"],
            map_location=DEVICE
        )
    )

    model.eval()
    models["U-Net"] = model

    # SegFormer
    model = SegFormer(num_classes=1).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINTS["SegFormer"],
            map_location=DEVICE
        )
    )

    model.eval()
    models["SegFormer"] = model

    # DeepLabV3+
    model = DeepLabV3Plus(num_classes=1).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINTS["DeepLabV3+"],
            map_location=DEVICE
        )
    )

    model.eval()
    models["DeepLabV3+"] = model

    # U-Net++
    model = UNetPlusPlus(num_classes=1).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINTS["U-Net++"],
            map_location=DEVICE
        )
    )

    model.eval()
    models["U-Net++"] = model

    print("All models loaded.")

    return models


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(prediction, target):

    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)

    tp = np.sum(
        (prediction == 1) &
        (target == 1)
    )

    fp = np.sum(
        (prediction == 1) &
        (target == 0)
    )

    fn = np.sum(
        (prediction == 0) &
        (target == 1)
    )

    tn = np.sum(
        (prediction == 0) &
        (target == 0)
    )

    # IoU
    union = tp + fp + fn

    if union == 0:
        iou = 1.0
    else:
        iou = tp / union

    # Dice
    dice_denominator = (
        2 * tp + fp + fn
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * tp /
            dice_denominator
        )

    # Precision
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = tp / precision_denominator

    # Recall
    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = tp / recall_denominator

    # Pixel accuracy
    total_pixels = tp + tn + fp + fn

    accuracy = (
        (tp + tn) /
        total_pixels
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


# ==========================================================
# RUN ONE MODEL
# ==========================================================

def predict(model, image):

    image = image.unsqueeze(0).to(DEVICE)

    logits = model(image)

    # Make sure output matches mask size
    logits = F.interpolate(
        logits,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    probability = torch.sigmoid(logits)

    prediction = (
        probability > 0.5
    ).float()

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return prediction


# ==========================================================
# CREATE QUALITATIVE FIGURE
# ==========================================================

def save_qualitative_case(
    case_number,
    test_index,
    image,
    target,
    predictions,
    metrics,
):

    image_np = (
        image
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    model_names = list(predictions.keys())

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(20, 4)
    )

    # Original
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(
        target,
        cmap="gray"
    )
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Four models
    for axis, model_name in zip(
        axes[2:],
        model_names
    ):

        axis.imshow(
            predictions[model_name],
            cmap="gray"
        )

        axis.set_title(
            f"{model_name}\n"
            f"IoU: {metrics[model_name]['iou']:.3f}"
        )

        axis.axis("off")

    fig.suptitle(
        f"Qualitative Case {case_number} "
        f"(Test Image {test_index:03d})",
        fontsize=14
    )

    plt.tight_layout()

    path = os.path.join(
        QUALITATIVE_DIR,
        f"case_{case_number:02d}_image_{test_index:03d}.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================================
# BAR CHART
# ==========================================================

def save_metric_plot(
    summary,
    metric,
    filename,
    title
):

    model_names = list(summary.keys())

    values = [
        summary[name][metric]
        for name in model_names
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        model_names,
        values
    )

    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(title)

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            filename
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 55)
    print("FINAL WILDFIRE MODEL ANALYSIS")
    print("=" * 55)

    print(
        f"Device: {DEVICE}"
    )

    # ------------------------------------------------------
    # Clean output
    # ------------------------------------------------------

    prepare_output_directory()

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE
    )

    print(
        f"Total samples: {len(dataset)}"
    )

    test_dataset = create_test_dataset(
        dataset
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # ------------------------------------------------------
    # Load models
    # ------------------------------------------------------

    models = load_models()

    model_names = list(models.keys())

    # ------------------------------------------------------
    # Store results
    # ------------------------------------------------------

    all_image_results = []

    # ------------------------------------------------------
    # Evaluate all test images
    # ------------------------------------------------------

    print("\nEvaluating all test images...")

    with torch.no_grad():

        for test_index in range(
            len(test_dataset)
        ):

            image, mask = test_dataset[
                test_index
            ]

            target = (
                mask
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            image_metrics = {}
            predictions = {}

            # ----------------------------------------------
            # Predict with all four models
            # ----------------------------------------------

            for model_name in model_names:

                prediction = predict(
                    models[model_name],
                    image
                )

                predictions[
                    model_name
                ] = prediction

                image_metrics[
                    model_name
                ] = calculate_metrics(
                    prediction,
                    target
                )

            # ----------------------------------------------
            # Find best and worst model
            # ----------------------------------------------

            ious = {
                name:
                image_metrics[name]["iou"]
                for name in model_names
            }

            best_model = max(
                ious,
                key=ious.get
            )

            worst_model = min(
                ious,
                key=ious.get
            )

            spread = (
                max(ious.values())
                -
                min(ious.values())
            )

            mean_iou = np.mean(
                list(ious.values())
            )

            all_image_results.append({
                "index": test_index + 1,
                "metrics": image_metrics,
                "best_model": best_model,
                "worst_model": worst_model,
                "spread": spread,
                "mean_iou": mean_iou,
                "image": image,
                "target": target,
                "predictions": predictions,
            })

            if (test_index + 1) % 20 == 0:

                print(
                    f"Processed "
                    f"{test_index + 1}/"
                    f"{len(test_dataset)}"
                )

    # ======================================================
    # QUANTITATIVE SUMMARY
    # ======================================================

    summary = {}

    for model_name in model_names:

        summary[model_name] = {}

        for metric in [
            "iou",
            "dice",
            "precision",
            "recall",
            "accuracy"
        ]:

            values = [
                item["metrics"][
                    model_name
                ][metric]

                for item in all_image_results
            ]

            summary[model_name][metric] = (
                float(np.mean(values))
            )

        summary[model_name]["parameters"] = sum(
            p.numel()
            for p in models[model_name].parameters()
        )

    # ======================================================
    # PRINT FINAL RESULTS
    # ======================================================

    print("\n")
    print("=" * 55)
    print("QUANTITATIVE RESULTS")
    print("=" * 55)

    print(
        f"{'Model':<15}"
        f"{'IoU':>10}"
        f"{'Dice':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
    )

    print("-" * 55)

    for model_name in model_names:

        print(
            f"{model_name:<15}"
            f"{summary[model_name]['iou']:>10.4f}"
            f"{summary[model_name]['dice']:>10.4f}"
            f"{summary[model_name]['precision']:>12.4f}"
            f"{summary[model_name]['recall']:>10.4f}"
        )

    print("\nParameter count:")

    for model_name in model_names:

        print(
            f"{model_name:<15}"
            f"{summary[model_name]['parameters']:,}"
        )

    # ======================================================
    # SAVE QUANTITATIVE PLOTS
    # ======================================================

    save_metric_plot(
        summary,
        "iou",
        "iou_comparison.png",
        "IoU Comparison"
    )

    save_metric_plot(
        summary,
        "dice",
        "dice_comparison.png",
        "Dice Comparison"
    )

    save_metric_plot(
        summary,
        "precision",
        "precision_comparison.png",
        "Precision Comparison"
    )

    save_metric_plot(
        summary,
        "recall",
        "recall_comparison.png",
        "Recall Comparison"
    )

    # ======================================================
    # SELECT QUALITATIVE CASES
    # ======================================================

    # Cases where models disagree most
    spread_cases = sorted(
        all_image_results,
        key=lambda item: item["spread"],
        reverse=True
    )

    # Hardest images for all models
    hard_cases = sorted(
        all_image_results,
        key=lambda item: item["mean_iou"]
    )

    selected = []

    # First: 6 largest differences
    for item in spread_cases:

        if len(selected) >= 6:
            break

        selected.append(item)

    # Then: 2 hardest cases
    for item in hard_cases:

        if len(selected) >= NUM_QUALITATIVE_CASES:
            break

        if item not in selected:
            selected.append(item)

    # ======================================================
    # SAVE QUALITATIVE FIGURES
    # ======================================================

    print("\nGenerating qualitative cases...")

    for case_number, item in enumerate(
        selected,
        start=1
    ):

        save_qualitative_case(
            case_number,
            item["index"],
            item["image"],
            item["target"],
            item["predictions"],
            item["metrics"],
        )

        print(
            f"Case {case_number}: "
            f"Test Image {item['index']:03d} | "
            f"Best: {item['best_model']} | "
            f"Worst: {item['worst_model']} | "
            f"Spread: {item['spread']:.3f}"
        )

    # ======================================================
    # FINISH
    # ======================================================

    print("\n")
    print("=" * 55)
    print("ANALYSIS COMPLETE")
    print("=" * 55)

    print(
        f"\nQuantitative plots:"
        f"\n{PLOT_DIR}"
    )

    print(
        f"\nQualitative cases:"
        f"\n{QUALITATIVE_DIR}"
    )


if __name__ == "__main__":
    main()