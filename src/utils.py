import random

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """AUC and LogLoss for a full split. y_prob must be probabilities (post-sigmoid), not raw logits."""
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob, labels=[0, 1]),
    }


class EarlyStopping:
    """Tracks the best value of a monitored metric and signals when to stop.

    Best-value tracking always runs regardless of `enabled`, so a run with
    early stopping disabled (e.g. for an overfitting-diagnosis ablation) can
    still report its best epoch/checkpoint without ever breaking the loop.
    """

    def __init__(self, patience: int, mode: str = "max", enabled: bool = True):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.enabled = enabled
        self.best = None
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float, epoch: int) -> bool:
        is_best = self.best is None or (
            value > self.best if self.mode == "max" else value < self.best
        )
        if is_best:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.enabled and self.counter >= self.patience:
                self.should_stop = True
        return is_best
