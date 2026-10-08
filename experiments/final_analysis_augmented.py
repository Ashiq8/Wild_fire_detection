import os
import csv

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.config import IMAGE_DIR, MASK_DIR, IMAGE_SIZE, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, DEVICE
from src.dataset import WildfireSegmentationDataset
from src.metrics import iou_score, dice_score
from experiments.unet_model import UNet
from src.model_unetpp import UNetPlusPlus
from experiments.model_segformer import SegFormer
from experiments.model_deeplab import DeepLabV3Plus
from experiments.attention_unet_model import AttentionUNet
from experiments.multiscale_unet_model import MultiScaleUNet

CHECKPOINTS = {
    "U-Net": "outputs/checkpoints/best_unet_aug.pth",
    "U-Net++": "outputs/checkpoints/best_unetpp_aug.pth",
    "Attention U-Net": "outputs/checkpoints/best_attention_unet.pth",
    "Multi-Scale U-Net": "outputs/checkpoints/best_multiscale_unet.pth",
    "SegFormer": "outputs/checkpoints/best_segformer_aug.pth",
    "DeepLabV3+": "outputs/checkpoints/best_deeplab_aug.pth",
}

OUTPUT_DIR = "outputs/final_analysis"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def create_test_loader(dataset):
    total_size = len(dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
def load_model(model_name):
    if model_name == "U-Net":
        model = UNet(num_classes=1)
    elif model_name == "U-Net++":
        model = UNetPlusPlus(num_classes=1)
    elif model_name == "Attention U-Net":
        model = AttentionUNet(num_classes=1)
    elif model_name == "Multi-Scale U-Net":
        model = MultiScaleUNet(num_classes=1)
    elif model_name == "SegFormer":
        model = SegFormer(num_classes=1)
    elif model_name == "DeepLabV3+":
        model = DeepLabV3Plus(num_classes=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = CHECKPOINTS[model_name]
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model
def evaluate_model(model, test_loader):
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            predictions = model(images)
            total_iou += iou_score(predictions, masks)
            total_dice += dice_score(predictions, masks)
            num_batches += 1
    return total_iou / num_batches, total_dice / num_batches


def save_csv(results):
    path = os.path.join(OUTPUT_DIR, "augmented_model_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Test IoU", "Test Dice"])
        for name, metrics in results.items():
            writer.writerow([name, f"{metrics['iou']:.4f}", f"{metrics['dice']:.4f}"])
    return path
def save_graph(results):
    names = list(results.keys())
    iou = [results[n]["iou"] for n in names]
    dice = [results[n]["dice"] for n in names]
    x = list(range(len(names)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], iou, width, label="Test IoU")
    bars2 = ax.bar([i + width/2 for i in x], dice, width, label="Test Dice")
    ax.set_title("Test Performance Comparison — Augmented Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "augmented_test_performance_comparison.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    return path


def main():
    print("=" * 60)
    print("AUGMENTED WILDFIRE MODEL EVALUATION")
    print("=" * 60)
    print("Device:", DEVICE)

    dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR, mask_dir=MASK_DIR, image_size=IMAGE_SIZE
    )
    print("Total samples:", len(dataset))

    test_loader = create_test_loader(dataset)
    print("Test samples:", len(test_loader.dataset))

    results = {}
    for model_name in CHECKPOINTS:
        print("\n" + "-" * 50)
        print("Evaluating:", model_name)
        model = load_model(model_name)
        test_iou, test_dice = evaluate_model(model, test_loader)
        results[model_name] = {"iou": test_iou, "dice": test_dice}
        print(f"Test IoU  : {test_iou:.4f}")
        print(f"Test Dice : {test_dice:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = save_csv(results)
    graph_path = save_graph(results)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Model':<25}{'Test IoU':>12}{'Test Dice':>12}")
    print("-" * 49)
    for name, metrics in results.items():
        print(f"{name:<25}{metrics['iou']:>12.4f}{metrics['dice']:>12.4f}")

    best = max(results, key=lambda n: results[n]["iou"])
    print("\nBest model:", best)
    print(f"Best Test IoU: {results[best]['iou']:.4f}")
    print(f"Best Test Dice: {results[best]['dice']:.4f}")
    print("\nSaved CSV:", csv_path)
    print("Saved graph:", graph_path)
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()