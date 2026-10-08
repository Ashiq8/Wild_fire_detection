import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import (
    IMAGE_DIR,
    MASK_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    NUM_CLASSES,
    RANDOM_SEED,
    DEVICE,
)

from src.dataset import (WildfireSegmentationDataset, get_train_transform)
from src.model_segformer import SegFormer
from src.losses import BCEDiceLoss
from src.metrics import dice_score, iou_score


# --------------------------------------------------
# Create output folders
# --------------------------------------------------

CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_DIR = Path("outputs/logs")


CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)



# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

def create_dataloaders(dataset):

    total_size = len(dataset)

    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(
        total_size,
        generator=generator
    ).tolist()

    train_indices = indices[:train_size]

    val_indices = indices[
        train_size:train_size + val_size
    ]

    test_indices = indices[
        train_size + val_size:
    ]

    train_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=256,
        transform=get_train_transform()
    )

    val_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=256,
        transform=None
    )

    test_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=256,
        transform=None
    )

    train_dataset.pairs = dataset.pairs
    val_dataset.pairs = dataset.pairs
    test_dataset.pairs = dataset.pairs

    train_dataset = torch.utils.data.Subset(
        train_dataset,
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        val_dataset,
        val_indices
    )

    test_dataset = torch.utils.data.Subset(
        test_dataset,
        test_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nDataset split:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# Train one epoch
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Calculate Dice loss
        loss = criterion(predictions, masks)

        # Autograd calculates gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            loss = criterion(predictions, masks)

            total_loss += loss.item()

            total_iou += iou_score(
                predictions,
                masks
            )

            total_dice += dice_score(
                predictions,
                masks
            )

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_iou, avg_dice


# --------------------------------------------------
# Test
# --------------------------------------------------

def test_model(model, loader):

    model.eval()

    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            total_iou += iou_score(
                predictions,
                masks
            )

            total_dice += dice_score(
                predictions,
                masks
            )

    avg_iou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_iou, avg_dice


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Device:", DEVICE)

    # Dataset
    dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=256
    )

    print("Total samples:", len(dataset))

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset
    )

    # Model
    print("\nLoading SegFormer-B0...")

    model = SegFormer(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print("SegFormer loaded.")

    # Loss
    criterion = BCEDiceLoss()

    # Optimizer
    # train_segformer.py
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

    # Parameter count
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # Best validation IoU
    best_val_iou = 0.0
    best_epoch = 0

    # History
    history = []

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_iou, val_dice = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        history.append([
            epoch + 1,
            train_loss,
            val_loss,
            val_iou,
            val_dice
        ])

        # Save best model
        if val_iou > best_val_iou:

            best_val_iou = val_iou
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_segformer_aug.pth"
            )

            print(
                f"  -> Best model saved "
                f"(Val IoU: {best_val_iou:.4f})"
            )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    history_file = LOG_DIR / "segformer_history.csv"

    with open(
        history_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_dice"
        ])

        writer.writerows(history)

    # --------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------

    print("\nLoading best SegFormer checkpoint...")

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_segformer.pth",
            map_location=DEVICE
        )
    )

    # --------------------------------------------------
    # Final test
    # --------------------------------------------------

    test_iou, test_dice = test_model(
        model,
        test_loader
    )

    print("\n==============================")
    print("SegFormer Final Test Results")
    print("==============================")
    print(f"Best Epoch : {best_epoch}")
    print(f"Best Val IoU : {best_val_iou:.4f}")
    print(f"Test IoU : {test_iou:.4f}")
    print(f"Test Dice : {test_dice:.4f}")
    print("==============================")

    print(
        "\nCheckpoint saved at:",
        CHECKPOINT_DIR / "best_segformer.pth"
    )

    print(
        "Training history saved at:",
        history_file
    )


if __name__ == "__main__":
    main()