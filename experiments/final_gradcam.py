from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import IMAGE_SIZE, DEVICE
from src.model_unetpp import UNetPlusPlus
from src.explainability.gradcam import SegmentationGradCAM


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path("data/images")
MASK_DIR = Path("data/masks")

CHECKPOINT = Path(
    "outputs/checkpoints/best_unetpp_aug.pth"
)

OUTPUT_DIR = Path(
    "outputs/explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Change this number to analyze another test image
TEST_INDEX = 0


# ============================================================
# FIND IMAGE
# ============================================================

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

if len(image_files) == 0:
    raise RuntimeError(
        "No images found in data/images"
    )

image_path = image_files[TEST_INDEX]


# ============================================================
# FIND MATCHING MASK
# ============================================================

# Your dataset matches images and masks using filename stem.
# Example:
# image_0.jpg -> image_0.png

mask_candidates = list(
    MASK_DIR.glob(
        f"{image_path.stem}.*"
    )
)

if len(mask_candidates) == 0:
    raise RuntimeError(
        f"No matching mask found for {image_path.name}"
    )

mask_path = mask_candidates[0]


print("================================")
print("Grad-CAM Explainability")
print("================================")
print("Device:", DEVICE)
print("Image :", image_path)
print("Mask  :", mask_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise RuntimeError(
        f"Could not read image: {image_path}"
    )

# BGR -> RGB
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# Resize exactly like your dataset
image = cv2.resize(
    image,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_LINEAR
)

# Normalize to [0, 1]
original = (
    image.astype(np.float32) / 255.0
)


# ============================================================
# LOAD GROUND-TRUTH MASK
# ============================================================

mask = cv2.imread(
    str(mask_path),
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise RuntimeError(
        f"Could not read mask: {mask_path}"
    )

# IMPORTANT:
# Nearest-neighbor for segmentation masks
mask = cv2.resize(
    mask,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_NEAREST
)

ground_truth = (
    mask > 0
).astype(np.float32)


# ============================================================
# CONVERT IMAGE TO PYTORCH TENSOR
# ============================================================

image_tensor = torch.from_numpy(
    original
).permute(
    2, 0, 1
).float()

image_input = image_tensor.unsqueeze(
    0
).to(DEVICE)

print(
    "Input shape:",
    tuple(image_input.shape)
)


# ============================================================
# LOAD U-NET++
# ============================================================

print("\nLoading U-Net++...")

model = UNetPlusPlus(
    num_classes=1
).to(DEVICE)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

if not CHECKPOINT.exists():

    raise RuntimeError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


# Support both common checkpoint formats
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print("U-Net++ loaded.")
print("Checkpoint:", CHECKPOINT)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating segmentation prediction...")

with torch.no_grad():

    logits = model(
        image_input
    )

    probabilities = torch.sigmoid(
        logits
    )

    prediction = (
        probabilities > 0.5
    ).float()


prediction = (
    prediction[0, 0]
    .cpu()
    .numpy()
)


print(
    "Predicted fire pixels:",
    int(prediction.sum())
)

print(
    "Ground-truth fire pixels:",
    int(ground_truth.sum())
)


# ============================================================
# GRAD-CAM
# ============================================================

# ============================================================
# GRAD-CAM
# ============================================================

print("\nGenerating Grad-CAM...")

target_layer = (
    model.model
    .decoder
    .blocks["x_0_4"]
    .conv2
)

gradcam = SegmentationGradCAM(
    model=model,
    target_layer=target_layer
)

# IMPORTANT:
# SegmentationGradCAM uses .generate()
# and returns both CAM and logits.
cam, gradcam_logits = gradcam.generate(
    image_input
)

# Convert tensor -> numpy if required
if torch.is_tensor(cam):

    cam = (
        cam
        .detach()
        .cpu()
        .numpy()
    )


# Remove unnecessary dimensions
cam = np.squeeze(
    cam
)


# Ensure values are [0, 1]
cam = np.clip(
    cam,
    0,
    1
)


# Resize Grad-CAM to original image size
cam = cv2.resize(
    cam,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_LINEAR
)


print(
    "Grad-CAM shape:",
    cam.shape
)


# ============================================================
# CREATE GRAD-CAM HEATMAP
# ============================================================

heatmap = cv2.applyColorMap(
    np.uint8(
        255 * cam
    ),
    cv2.COLORMAP_JET
)

# OpenCV BGR -> RGB
heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)

heatmap = (
    heatmap.astype(np.float32)
    / 255.0
)


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = (
    0.55 * original
    +
    0.45 * heatmap
)

overlay = np.clip(
    overlay,
    0,
    1
)


# ============================================================
# OUTPUT FILE PREFIX
# ============================================================

prefix = (
    f"sample_{TEST_INDEX:04d}"
)


# ============================================================
# SAVE ORIGINAL
# ============================================================

original_path = (
    OUTPUT_DIR /
    f"{prefix}_original.png"
)

plt.imsave(
    original_path,
    original
)


# ============================================================
# SAVE GROUND TRUTH
# ============================================================

ground_truth_path = (
    OUTPUT_DIR /
    f"{prefix}_ground_truth.png"
)

plt.imsave(
    ground_truth_path,
    ground_truth,
    cmap="gray"
)


# ============================================================
# SAVE PREDICTION
# ============================================================

prediction_path = (
    OUTPUT_DIR /
    f"{prefix}_prediction.png"
)

plt.imsave(
    prediction_path,
    prediction,
    cmap="gray"
)


# ============================================================
# SAVE GRAD-CAM
# ============================================================

gradcam_path = (
    OUTPUT_DIR /
    f"{prefix}_gradcam.png"
)

plt.imsave(
    gradcam_path,
    cam,
    cmap="jet"
)


# ============================================================
# SAVE OVERLAY
# ============================================================

overlay_path = (
    OUTPUT_DIR /
    f"{prefix}_overlay.png"
)

plt.imsave(
    overlay_path,
    overlay
)


# ============================================================
# FINAL EXPLAINABILITY PANEL
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(20, 4)
)


# ------------------------------------------------------------
# 1. Original
# ------------------------------------------------------------

axes[0].imshow(
    original
)

axes[0].set_title(
    "Original Image"
)


# ------------------------------------------------------------
# 2. Ground Truth
# ------------------------------------------------------------

axes[1].imshow(
    ground_truth,
    cmap="gray"
)

axes[1].set_title(
    "Ground Truth"
)


# ------------------------------------------------------------
# 3. Prediction
# ------------------------------------------------------------

axes[2].imshow(
    prediction,
    cmap="gray"
)

axes[2].set_title(
    "U-Net++ Prediction"
)


# ------------------------------------------------------------
# 4. Grad-CAM
# ------------------------------------------------------------

axes[3].imshow(
    cam,
    cmap="jet"
)

axes[3].set_title(
    "Grad-CAM"
)


# ------------------------------------------------------------
# 5. Overlay
# ------------------------------------------------------------

axes[4].imshow(
    overlay
)

axes[4].set_title(
    "Grad-CAM Overlay"
)


# Remove axes
for ax in axes:

    ax.axis("off")


plt.tight_layout()


# ============================================================
# SAVE FINAL PANEL
# ============================================================

panel_path = (
    OUTPUT_DIR /
    f"{prefix}_explainability_panel.png"
)

plt.savefig(
    panel_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# REMOVE GRAD-CAM HOOKS
# ============================================================

gradcam.remove_hooks()


# ============================================================
# COMPLETE
# ============================================================

print("\n================================")
print("GRAD-CAM COMPLETE")
print("================================")

print(
    "Original      :",
    original_path
)

print(
    "Ground Truth  :",
    ground_truth_path
)

print(
    "Prediction    :",
    prediction_path
)

print(
    "Grad-CAM      :",
    gradcam_path
)

print(
    "Overlay       :",
    overlay_path
)

print(
    "Final Panel   :",
    panel_path
)

print("\nDone bro.")