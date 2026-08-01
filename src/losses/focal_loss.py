"""Binary Focal Loss, used to address the extreme class imbalance (~1:1000) in CTR prediction.

Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
where p_t is the model's predicted probability for the true class.
Compared to standard BCE, the (1 - p_t)^gamma term down-weights "easy" samples
(the large number of easy negatives), letting the model focus more on hard
samples. A larger gamma suppresses easy samples more strongly; gamma=0
reduces to standard (weighted) BCE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: raw model scores (pre-sigmoid); targets: 0/1 labels."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compare_with_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25):
    """Returns (bce_loss, focal_loss) for side-by-side comparison / plotting."""
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets).item()
    focal_loss = FocalLoss(gamma=gamma, alpha=alpha)(logits, targets).item()
    return bce_loss, focal_loss


if __name__ == "__main__":
    torch.manual_seed(0)

    # Simulate an extreme imbalance scenario: 1000 samples, ~1/1000 positive
    n = 1000
    targets = torch.zeros(n)
    targets[:1] = 1  # 1 positive sample

    # Scenario: the model is already confident on most "easy negatives"
    # (very negative logit), with only a single hard sample.
    easy_neg_logits = torch.full((n,), -4.0)
    hard_logit = torch.tensor([0.5])  # not confident enough on the one positive
    logits = easy_neg_logits.clone()
    logits[:1] = hard_logit

    bce, focal = compare_with_bce(logits, targets)
    print(f"BCE loss  : {bce:.6f}")
    print(f"Focal loss: {focal:.6f}  (gamma=2.0, alpha=0.25)")
    print("Focal loss should be significantly lower than BCE, since the (1-p_t)^gamma "
          "term suppresses the contribution of the many easy negatives.")
