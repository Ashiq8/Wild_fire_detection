import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation


class SegFormer(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # ImageNet normalization for pretrained SegFormer
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):

        # Dataset gives image values in [0, 1]
        x = (x - self.mean) / self.std

        # SegFormer forward pass
        outputs = self.model(pixel_values=x)

        # Resize output to original image size
        logits = F.interpolate(
            outputs.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return logits