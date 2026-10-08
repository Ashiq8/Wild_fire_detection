import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import (
    IMAGE_DIR,
    MASK_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    DEVICE,
)

from src.dataset import (
    WildfireSegmentationDataset,
    get_train_transform
)

from experiments.attention_unet_model import AttentionUNet
from src.losses import BCEDiceLoss
from src.metrics import iou_score, dice_score


CHECKPOINT_PATH = Path(
    "outputs/checkpoints/best_attention_unet.pth"
)


def create_dataloaders(dataset):

    total_size = len(dataset)

    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

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

    # Training dataset → augmentation ON
    train_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        transform=get_train_transform()
    )

    # Validation dataset → augmentation OFF
    val_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        transform=None
    )

    # Test dataset → augmentation OFF
    test_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        transform=None
    )

    # Use the same image-mask pairs
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

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            masks
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            loss = criterion(
                predictions,
                masks
            )

            total_loss += loss.item()

            total_iou += iou_score(
                predictions,
                masks
            )

            total_dice += dice_score(
                predictions,
                masks
            )

    return (
        total_loss / len(loader),
        total_iou / len(loader),
        total_dice / len(loader)
    )


def test_model(
    model,
    loader
):

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

    return (
        total_iou / len(loader),
        total_dice / len(loader)
    )


def main():

    print("Using device:", DEVICE)

    # Dataset
    dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE
    )

    print("Total samples:", len(dataset))

    # DataLoaders
    train_loader, val_loader, test_loader = (
        create_dataloaders(dataset)
    )

    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    # Model
    print("\nLoading Attention U-Net...")

    model = AttentionUNet(
        num_classes=1
    ).to(DEVICE)

    print("Attention U-Net loaded.")

    # Parameters
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        total_params
    )

    print(
        "Trainable parameters:",
        trainable_params
    )

    # Loss
    criterion = BCEDiceLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_iou = 0.0
    best_epoch = 0

    # Training
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

        if val_iou > best_val_iou:

            best_val_iou = val_iou
            best_epoch = epoch + 1

            CHECKPOINT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print(
                f"  -> Best model saved "
                f"(Val IoU: {best_val_iou:.4f})"
            )

    # Load best checkpoint
    print("\nLoading best Attention U-Net checkpoint...")

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )
    )

    # Final test
    test_iou, test_dice = test_model(
        model,
        test_loader
    )

    print("\n==============================")
    print("Attention U-Net Final Results")
    print("==============================")

    print(
        f"Best Epoch : {best_epoch}"
    )

    print(
        f"Best Val IoU : {best_val_iou:.4f}"
    )

    print(
        f"Test IoU : {test_iou:.4f}"
    )

    print(
        f"Test Dice : {test_dice:.4f}"
    )

    print("==============================")


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    