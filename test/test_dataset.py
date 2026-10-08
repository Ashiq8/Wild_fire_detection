from src.config import IMAGE_DIR, MASK_DIR, IMAGE_SIZE
from src.dataset import WildfireSegmentationDataset


dataset =  WildfireSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE
)

print("\nDataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape :", mask.shape)

print("Image dtype:", image.dtype)
print("Mask dtype :", mask.dtype)

print("Image min  :", image.min().item())
print("Image max  :", image.max().item())

print("Mask values:", mask.unique().tolist())