import torch
from torch.utils.data import DataLoader

from src.config import IMAGE_DIR, MASK_DIR, IMAGE_SIZE, BATCH_SIZE
from src.dataset import WildfireSegmentationDataset


dataset = WildfireSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

images, masks = next(iter(loader))

print("Dataset size :", len(dataset))
print("Images shape :", images.shape)
print("Masks shape  :", masks.shape)
print("Image dtype  :", images.dtype)
print("Mask dtype   :", masks.dtype)
print("Image range  :", images.min().item(), images.max().item())
print("Mask values  :", torch.unique(masks).tolist())