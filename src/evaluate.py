import torch
from torch.utils.data import DataLoader

from src.config import (
    IMAGE_DIR,
    MASK_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    DEVICE,
)

from src.dataset import WildfireSegmentationDataset
from src.model_unetpp import UNetPlusPlus
from src.metrics import iou_score, dice_score


def create_test_loader():

    dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        transform=None
    )

    total_size = len(dataset)

    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(
        total_size,
        generator=generator
    ).tolist()

    test_indices = indices[train_size + val_size:]

    test_dataset = torch.utils.data.Subset(
        dataset,
        test_indices
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return test_loader


def main():

    print("Using device:", DEVICE)

    test_loader = create_test_loader()

    model = UNetPlusPlus(num_classes=1).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "outputs/checkpoints/best_unetpp_aug.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            total_iou += iou_score(predictions, masks)
            total_dice += dice_score(predictions, masks)

    avg_iou = total_iou / len(test_loader)
    avg_dice = total_dice / len(test_loader)

    print("\nTest Results")
    print("------------")
    print(f"Test IoU:  {avg_iou:.4f}")
    print(f"Test Dice: {avg_dice:.4f}")


if __name__ == "__main__":
    main()