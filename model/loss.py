"""Loss implementations."""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss class for image segmentation tasks."""

    def __init__(
        self,
        gamma: float = 2,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        """Focal loss for multi class.

        Args:
            gamma: Focusing parameter
            reduction: Specifies the reduction to apply to the output

        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduction

    def forward(self, x, y):
        """Forward pass."""
        if x.dim() > 2:
            x = torch.einsum("ijkl->iklj", x)  # N,C,H,W => N,H,W,C
            x = x.contiguous().view(-1, x.shape[-1])  # N,C,H,W => N*H*W,C

        y = y.view(-1)

        logpt = -F.cross_entropy(x, y, reduction="none")
        pt = torch.exp(logpt)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduce == "mean":
            return loss.mean()
        elif self.reduce == "sum":
            return loss.sum()
        else:
            return ValueError(f"{self.reduce} is not in ['mean', 'sum']")
