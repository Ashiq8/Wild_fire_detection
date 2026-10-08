import torch
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

from src.dataset import WildfireSegmentationDataset, get_train_transform
from src.unet_model import UNet 
from src.losses import BCEDiceLoss
from src.metrics import iou_score, dice_score


def create_dataloaders(dataset):

    total_size = len(dataset)

    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    # Create the same fixed split every time
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create separate dataset objects
    train_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        transform=get_train_transform()
    )

    val_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        transform=None
    )

    test_dataset = WildfireSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        transform=None
    )

    # Make sure all three use the exact same image ordering
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

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(predictions, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


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
            total_iou += iou_score(predictions, masks)
            total_dice += dice_score(predictions, masks)

    return (
        total_loss / len(loader),
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

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset)

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Model
    model = UNet(num_classes=1).to(DEVICE)

    # Loss
    criterion = BCEDiceLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_iou = 0.0

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
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val IoU: {val_iou:.4f} "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_iou > best_val_iou:

            best_val_iou = val_iou

            torch.save(
                model.state_dict(),
                "outputs/checkpoints/best_unet_aug.pth"
            )

            print("Best model saved!")


if __name__ == "__main__":
    main()