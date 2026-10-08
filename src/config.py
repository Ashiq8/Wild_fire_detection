import torch
from pathlib import Path

IMAGE_DIR = Path("data/images")
MASK_DIR = Path("data/masks")

IMAGE_SIZE = 256

BATCH_SIZE = 8

NUM_EPOCHS = 30

LEARNING_RATE = 0.0001

TRAIN_RATIO = 0.80

VAL_RATIO = 0.10

TEST_RATIO = 0.10

NUM_CLASSES = 1

RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

