"""Segmentation metrics."""

import torch
from torchmetrics import Dice


def compute_accuracy(
    label: torch.Tensor, predicted: torch.Tensor
) -> torch.Tensor:
    """Compute segmentation accuracy."""
    acc = (
        label.cpu() == torch.argmax(predicted, axis=1).cpu()
    ).sum() / torch.numel(label.cpu())
    return acc


def compute_dice_score(
    mask: torch.Tensor, gt_mask: torch.Tensor
) -> torch.Tensor:
    """Compute segmentation dice score."""
    dice_score = Dice(average="micro")
    return dice_score(mask, gt_mask)
