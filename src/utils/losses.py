import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        gamma: focusing parameter (typically 1.0 - 3.0)
        alpha: class weights tensor of shape (C,) or None
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw model outputs of shape (N, C)
            targets: ground truth labels of shape (N,)
        """
        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Gather log_probs and probs for the ground-truth class
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        # Focal term
        focal_term = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                alpha = self.alpha.to(logits.device)
            else:
                alpha = self.alpha
            at = alpha.gather(0, targets.squeeze(1))
            loss = -at * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss