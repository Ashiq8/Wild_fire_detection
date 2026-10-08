import cv2
import numpy as np
from pathlib import Path
import random
import csv


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = Path("data/images")
MASK_DIR = Path("data/masks")

OUTPUT_DIR = Path("dataset_check")
OVERLAY_DIR = OUTPUT_DIR / "overlays"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND FILES
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png"}
mask_extensions = {".png", ".jpg", ".jpeg"}

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

mask_files = [
    p for p in MASK_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in mask_extensions
]


# ============================================================
# CREATE LOOKUP TABLES
# ============================================================
 
images = {
    p.stem: p
    for p in image_files
}

masks = {
    p.stem: p
    for p in mask_files
}


image_names = set(images.keys())
mask_names = set(masks.keys())


matched_names = sorted(image_names & mask_names)
images_without_masks = sorted(image_names - mask_names)
masks_without_images = sorted(mask_names - image_names)


# ============================================================
# BASIC COUNTS
# ============================================================

print("\n" + "=" * 60)
print("FLAME DATASET VERIFICATION")
print("=" * 60)

print(f"Total images          : {len(image_files)}")
print(f"Total masks           : {len(mask_files)}")
print(f"Matched pairs         : {len(matched_names)}")
print(f"Images without masks  : {len(images_without_masks)}")
print(f"Masks without images  : {len(masks_without_images)}")


# ============================================================
# CHECK MATCHED PAIRS
# ============================================================

valid_pairs = []
invalid_pairs = []

empty_masks = []
unusual_masks = []


print("\nChecking image-mask pairs...\n")


for i, name in enumerate(matched_names, start=1):

    image_path = images[name]
    mask_path = masks[name]

    image = cv2.imread(str(image_path))
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        invalid_pairs.append(
            (name, "image could not be read")
        )
        continue

    if mask is None:
        invalid_pairs.append(
            (name, "mask could not be read")
        )
        continue


    # --------------------------------------------------------
    # SIZE CHECK
    # --------------------------------------------------------

    image_height, image_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    if (
        image_height != mask_height
        or image_width != mask_width
    ):

        invalid_pairs.append(
            (
                name,
                f"size mismatch: "
                f"image={image_width}x{image_height}, "
                f"mask={mask_width}x{mask_height}"
            )
        )

        continue


    # --------------------------------------------------------
    # MASK VALUES
    # --------------------------------------------------------

    unique_values = np.unique(mask)

    foreground_pixels = np.count_nonzero(mask)

    total_pixels = mask.shape[0] * mask.shape[1]

    foreground_ratio = (
        foreground_pixels / total_pixels
    )


    # --------------------------------------------------------
    # EMPTY MASK
    # --------------------------------------------------------

    if foreground_pixels == 0:

        empty_masks.append(name)


    # --------------------------------------------------------
    # UNUSUAL VALUES
    # --------------------------------------------------------

    # Common binary masks are:
    # 0 / 1
    # or
    # 0 / 255

    allowed_common = {0, 1, 255}

    if not set(unique_values).issubset(
        allowed_common
    ):

        unusual_masks.append(
            (
                name,
                unique_values.tolist()
            )
        )


    # --------------------------------------------------------
    # STORE VALID PAIR
    # --------------------------------------------------------

    valid_pairs.append(
        {
            "name": name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "width": image_width,
            "height": image_height,
            "unique_values": unique_values.tolist(),
            "foreground_pixels": int(
                foreground_pixels
            ),
            "foreground_ratio": foreground_ratio
        }
    )


    # Progress
    if i % 100 == 0:
        print(
            f"Checked {i}/{len(matched_names)} pairs..."
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CHECK RESULTS")
print("=" * 60)

print(f"Valid pairs           : {len(valid_pairs)}")
print(f"Invalid pairs         : {len(invalid_pairs)}")
print(f"Empty masks           : {len(empty_masks)}")
print(f"Unusual mask values   : {len(unusual_masks)}")


# ============================================================
# SHOW SOME MASK VALUE EXAMPLES
# ============================================================

print("\nExample mask values:")

for item in valid_pairs[:10]:

    print(
        f"{item['name']}: "
        f"{item['unique_values']}"
    )


# ============================================================
# SAVE CSV REPORT
# ============================================================

csv_path = OUTPUT_DIR / "dataset_report.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "name",
            "image_path",
            "mask_path",
            "width",
            "height",
            "unique_values",
            "foreground_pixels",
            "foreground_ratio"
        ]
    )

    writer.writeheader()

    writer.writerows(valid_pairs)


# ============================================================
# SAVE PROBLEM LISTS
# ============================================================

with open(
    OUTPUT_DIR / "images_without_masks.txt",
    "w",
    encoding="utf-8"
) as file:

    for name in images_without_masks:
        file.write(name + "\n")


with open(
    OUTPUT_DIR / "masks_without_images.txt",
    "w",
    encoding="utf-8"
) as file:

    for name in masks_without_images:
        file.write(name + "\n")


with open(
    OUTPUT_DIR / "empty_masks.txt",
    "w",
    encoding="utf-8"
) as file:

    for name in empty_masks:
        file.write(name + "\n")


# ============================================================
# VISUAL CHECK
# ============================================================

print("\nCreating visual samples...")


# Select up to 10 random pairs
sample_count = min(10, len(valid_pairs))

if sample_count > 0:

    random.seed(42)

    samples = random.sample(
        valid_pairs,
        sample_count
    )

    for item in samples:

        image = cv2.imread(
            item["image_path"]
        )

        mask = cv2.imread(
            item["mask_path"],
            cv2.IMREAD_GRAYSCALE
        )


        # ----------------------------------------------------
        # Convert mask to binary
        # ----------------------------------------------------

        binary_mask = np.where(
            mask > 0,
            255,
            0
        ).astype(np.uint8)


        # ----------------------------------------------------
        # Create red overlay
        # ----------------------------------------------------

        overlay = image.copy()

        overlay[binary_mask == 255] = (
            0,
            0,
            255
        )


        result = cv2.addWeighted(
            image,
            0.7,
            overlay,
            0.3,
            0
        )


        # ----------------------------------------------------
        # Resize for easy viewing
        # ----------------------------------------------------

        max_width = 1000
        max_height = 700

        height, width = result.shape[:2]

        scale = min(
            max_width / width,
            max_height / height,
            1.0
        )

        if scale < 1:

            result = cv2.resize(
                result,
                (
                    int(width * scale),
                    int(height * scale)
                )
            )


        # ----------------------------------------------------
        # Save overlay
        # ----------------------------------------------------

        output_path = (
            OVERLAY_DIR /
            f"{item['name']}_overlay.jpg"
        )

        cv2.imwrite(
            str(output_path),
            result
        )


print(
    f"\nVisual samples saved to: "
    f"{OVERLAY_DIR.resolve()}"
)


# ============================================================
# PRINT PROBLEMS
# ============================================================

if invalid_pairs:

    print("\nInvalid pairs:")

    for item in invalid_pairs[:20]:
        print(" ", item)


if empty_masks:

    print("\nFirst empty masks:")

    for name in empty_masks[:20]:
        print(" ", name)


if unusual_masks:

    print("\nUnusual mask values:")

    for item in unusual_masks[:20]:
        print(" ", item)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Images              : {len(image_files)}")
print(f"Masks               : {len(mask_files)}")
print(f"Matched             : {len(matched_names)}")
print(f"Valid               : {len(valid_pairs)}")
print(f"Invalid             : {len(invalid_pairs)}")
print(f"Empty masks         : {len(empty_masks)}")
print(f"Images without mask : {len(images_without_masks)}")
print(f"Masks without image : {len(masks_without_images)}")

print("\nReport saved to:")
print(csv_path.resolve())

print("\nDone.")