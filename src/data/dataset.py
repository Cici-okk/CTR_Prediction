"""CTRDataset plus sampling strategies for extreme class imbalance (~1:1000).

Supports two strategies (matching sampling.strategy in configs/config.yaml):
  - class_weight: keeps the data distribution unchanged, and instead uses a
    WeightedRandomSampler (or loss weighting) during training
  - undersample:  undersamples negatives to bring neg:pos down to a target
    ratio (e.g. 10:1), speeding up training
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class CTRDataset(Dataset):
    def __init__(self, parquet_path: str, meta: dict):
        df = pd.read_parquet(parquet_path)
        self.label_col = meta["label_col"]
        self.numeric_cols = meta["numeric_cols"]
        self.categorical_cols = meta["categorical_cols"]

        self.labels = torch.tensor(df[self.label_col].values, dtype=torch.float32)
        self.numeric = torch.tensor(
            df[self.numeric_cols].values, dtype=torch.float32
        ) if self.numeric_cols else torch.zeros((len(df), 0))
        self.categorical = torch.tensor(
            df[self.categorical_cols].values, dtype=torch.long
        ) if self.categorical_cols else torch.zeros((len(df), 0), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "numeric": self.numeric[idx],
            "categorical": self.categorical[idx],
            "label": self.labels[idx],
        }


def undersample_negatives(df: pd.DataFrame, label_col: str, neg_ratio: float, seed: int) -> pd.DataFrame:
    """Undersample negatives down to neg_ratio * (number of positives); all positives are kept."""
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]
    n_neg_keep = min(len(neg_df), int(len(pos_df) * neg_ratio))
    neg_sampled = neg_df.sample(n=n_neg_keep, random_state=seed)
    out = pd.concat([pos_df, neg_sampled], axis=0).sample(frac=1, random_state=seed)
    return out.reset_index(drop=True)


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Compute an inverse-frequency sampling weight per sample, for use with WeightedRandomSampler."""
    pos = labels.sum().item()
    neg = len(labels) - pos
    weight_pos = 1.0 / max(pos, 1)
    weight_neg = 1.0 / max(neg, 1)
    return torch.where(labels == 1, weight_pos, weight_neg)


def build_dataloader(
    processed_dir: str,
    split: str,
    batch_size: int,
    strategy: str = "class_weight",
    undersample_neg_ratio: float = 10.0,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    with open(os.path.join(processed_dir, "meta.json")) as f:
        meta = json.load(f)

    parquet_path = os.path.join(processed_dir, f"{split}.parquet")

    if split == "train" and strategy == "undersample":
        df = pd.read_parquet(parquet_path)
        df = undersample_negatives(df, meta["label_col"], undersample_neg_ratio, seed)
        tmp_path = os.path.join(processed_dir, "_train_undersampled.parquet")
        df.to_parquet(tmp_path, index=False)
        parquet_path = tmp_path

    dataset = CTRDataset(parquet_path, meta)

    if split == "train" and strategy == "class_weight":
        weights = compute_class_weights(dataset.labels)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and split == "train"))
