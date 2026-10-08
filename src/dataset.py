from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A


class WildfireSegmentationDataset(Dataset):
    """
    Wildfire semantic segmentation dataset.

    Returns:
        image: Float tensor [C, H, W] in [0, 1]
        mask:  Float tensor [1, H, W] with values 0 or 1

    Augmentation is applied only when transform is provided.
    """

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    }

    MASK_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
    }

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=256,
        transform=None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.transform = transform

        self.pairs = self._build_pairs()

        if not self.pairs:
            raise RuntimeError(
                "No valid image-mask pairs found."
            )

    def _build_pairs(self):
        """Match images and masks using filename stem."""

        image_files = {
            path.stem: path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.IMAGE_EXTENSIONS
        }

        mask_files = {
            path.stem: path
            for path in self.mask_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.MASK_EXTENSIONS
        }

        common_stems = sorted(
            set(image_files) & set(mask_files)
        )

        pairs = [
            (image_files[stem], mask_files[stem])
            for stem in common_stems
        ]

        print(f"Images found : {len(image_files)}")
        print(f"Masks found  : {len(mask_files)}")
        print(f"Valid pairs  : {len(pairs)}")

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

       
        # Read image
        
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        
        # Read mask
        
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

       
        # Resize before augmentation
        
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Apply augmentation
       
        if self.transform is not None:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        
        # Image -> [0, 1]
        
        image = (
            image.astype(np.float32) / 255.0
        )

        
        # Binary mask
        
        mask = (
            mask > 0
        ).astype(np.float32)

       
        # HWC -> CHW
        
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        
        # Mask: H,W -> 1,H,W
       
        mask = np.expand_dims(
            mask,
            axis=0
        )

        
        # NumPy -> PyTorch
        
        image = torch.from_numpy(
            np.ascontiguousarray(image)
        )

        mask = torch.from_numpy(
            np.ascontiguousarray(mask)
        )

        return image, mask



# Training augmentation


def get_train_transform(image_size=256):

    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),

        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.3
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),

        A.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02,
            p=0.2
        ),
    ])