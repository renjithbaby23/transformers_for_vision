"""Loss implementations."""
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """Focal loss class for image segmentation tasks."""

    def __init__(
        self,
        gamma: float = 2,
        alpha: Optional[list] = None,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        """Focal loss for multi class.

        Args:
            gamma: Focusing parameter
            alpha: Scaling parameter
            reduction: Specifies the reduction to apply to the output

        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduction
        self.alpha: Optional[torch.Tensor] = (
            torch.tensor(alpha) if alpha is not None else None
        )

    def forward(self, x, y):
        """Forward pass."""
        if x.dim() > 2:
            x = torch.einsum("ijkl->iklj", x)  # N,C,H,W => N,H,W,C
            x = x.contiguous().view(-1, x.shape[-1])  # N,C,H,W => N*H*W,C

        y = y.view(-1)

        logpt = -F.cross_entropy(x, y, reduction="none")
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.type_as(x.data)
            at = self.alpha.gather(0, y.data)
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduce == "mean":
            return loss.mean()
        elif self.reduce == "sum":
            return loss.sum()
        else:
            return ValueError(f"{self.reduce} is not in ['mean', 'sum']")
