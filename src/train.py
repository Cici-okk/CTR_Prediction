"""Training script for the FM baseline: BCE vs Focal loss, Adam vs AdamW,
Dropout + Early Stopping + LR Warmup, AUC/LogLoss tracking, Train/Val gap
diagnosis.

Usage:
    python -m src.train --output_dir runs/smoke --epochs 1
    python -m src.train --loss focal --optimizer adamw --output_dir runs/focal_adamw
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.dataset import build_dataloader
from src.losses.focal_loss import FocalLoss
from src.models.fm import FactorizationMachine
from src.models.transformer import TransformerCTR
from src.utils import EarlyStopping, compute_metrics, set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default="runs/default")
    parser.add_argument("--model_type", type=str, choices=["fm", "transformer"], default=None)
    parser.add_argument("--loss", type=str, choices=["bce", "focal"], default=None)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--embedding_weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(meta: dict, cfg: dict) -> nn.Module:
    model_type = cfg["model"].get("type", "fm")
    if model_type == "transformer":
        transformer_kwargs = cfg["model"].get("transformer", {})
        return TransformerCTR(
            meta["vocab_sizes"], meta["categorical_cols"],
            embed_dim=cfg["model"]["embed_dim"], dropout=cfg["model"]["dropout"],
            **transformer_kwargs,
        )
    return FactorizationMachine(
        meta["vocab_sizes"], meta["categorical_cols"],
        embed_dim=cfg["model"]["embed_dim"], dropout=cfg["model"]["dropout"],
    )


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Splits params into the shared field-embedding table vs everything else, so the
    embedding table (dominated by high-cardinality, often near-unique-per-row fields like
    device_ip) can be regularized independently of e.g. attention/FFN weights. Defaults to
    the same weight_decay for both groups when `embedding_weight_decay` isn't set, which is
    numerically identical to a single param group - existing configs are unaffected."""
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    embedding_weight_decay = float(cfg["train"].get("embedding_weight_decay", weight_decay))

    embedding_params, other_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("embedding."):
            embedding_params.append(param)
        else:
            other_params.append(param)
    param_groups = [
        {"params": embedding_params, "weight_decay": embedding_weight_decay},
        {"params": other_params, "weight_decay": weight_decay},
    ]

    if cfg["train"]["optimizer"] == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    return torch.optim.AdamW(param_groups, lr=lr)


def build_criterion(cfg: dict) -> nn.Module:
    if cfg["loss"]["type"] == "bce":
        return nn.BCEWithLogitsLoss()
    return FocalLoss(gamma=cfg["loss"]["focal_gamma"], alpha=cfg["loss"]["focal_alpha"])


def build_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    warmup_steps = max(warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    probs_all, labels_all = [], []
    for batch in loader:
        categorical = batch["categorical"].to(device)
        labels = batch["label"].to(device)
        logits = model(categorical)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    metrics = compute_metrics(labels_all, probs_all)
    metrics["loss"] = total_loss / n
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        categorical = batch["categorical"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(categorical)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_type is not None:
        cfg["model"]["type"] = args.model_type
    if args.loss is not None:
        cfg["loss"]["type"] = args.loss
    if args.optimizer is not None:
        cfg["train"]["optimizer"] = args.optimizer
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.embedding_weight_decay is not None:
        cfg["train"]["embedding_weight_decay"] = args.embedding_weight_decay
    if args.epochs is not None:
        cfg["train"]["num_epochs"] = args.epochs

    set_seed(cfg["data"]["random_seed"])
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    processed_dir = cfg["data"]["processed_dir"]
    with open(os.path.join(processed_dir, "meta.json")) as f:
        meta = json.load(f)

    batch_size = cfg["train"]["batch_size"]
    train_loader = build_dataloader(
        processed_dir, "train", batch_size,
        strategy=cfg["sampling"]["strategy"],
        undersample_neg_ratio=cfg["sampling"]["undersample_neg_ratio"],
        seed=cfg["data"]["random_seed"],
    )
    # Separate, unsampled pass over the same training rows: used only for the
    # clean train-side AUC/LogLoss diagnostic, so it's comparable to val/test
    # (which see the natural, non-rebalanced positive rate).
    train_eval_loader = build_dataloader(
        processed_dir, "train", batch_size, strategy="none", shuffle=False,
    )
    val_loader = build_dataloader(processed_dir, "val", batch_size, strategy="none", shuffle=False)
    test_loader = build_dataloader(processed_dir, "test", batch_size, strategy="none", shuffle=False)

    model = build_model(meta, cfg).to(device)

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_warmup_scheduler(optimizer, cfg["train"]["warmup_steps"])

    metric_name = cfg["train"]["early_stopping_metric"]
    mode = "max" if metric_name == "auc" else "min"
    early_stopper = EarlyStopping(
        patience=cfg["train"]["early_stopping_patience"],
        mode=mode,
        enabled=not args.no_early_stop,
    )

    best_state = None
    history = []

    for epoch in range(cfg["train"]["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_metrics = evaluate(model, train_eval_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_metrics["auc"],
            "train_logloss": train_metrics["logloss"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_logloss": val_metrics["logloss"],
            "gap": train_metrics["auc"] - val_metrics["auc"],
        }
        history.append(record)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"train_logloss={train_metrics['logloss']:.4f} val_logloss={val_metrics['logloss']:.4f} "
            f"gap={record['gap']:.4f}"
        )

        monitored = val_metrics[metric_name]
        is_best = early_stopper.step(monitored, epoch)
        if is_best:
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best epoch {early_stopper.best_epoch})")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_epoch": early_stopper.best_epoch,
        "val_auc": early_stopper.best if metric_name == "auc" else history[early_stopper.best_epoch]["val_auc"],
        "val_logloss": history[early_stopper.best_epoch]["val_logloss"],
        "test_auc": test_metrics["auc"],
        "test_logloss": test_metrics["logloss"],
        "loss_type": cfg["loss"]["type"],
        "optimizer": cfg["train"]["optimizer"],
        "dropout": cfg["model"]["dropout"],
        "early_stopping_enabled": not args.no_early_stop,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Best epoch {summary['best_epoch']}: test_auc={test_metrics['auc']:.4f} test_logloss={test_metrics['logloss']:.4f}")


if __name__ == "__main__":
    main()
