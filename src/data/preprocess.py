"""Data preprocessing pipeline: read raw CTR data -> handle missing values ->
encode categorical features -> split into train/val/test.

Usage:
    python -m src.data.preprocess --raw_path data/raw/train.csv --out_dir data/processed
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_missing(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("__missing__").astype(str)
    return df


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical features; returns the encoded df and each column's vocab size (for embedding layers)."""
    vocab_sizes = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        vocab_sizes[col] = len(le.classes_)
    return df, vocab_sizes


def split_data(
    df: pd.DataFrame, label_col: str, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split, keeping the positive ratio consistent across train/val/test under extreme imbalance."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val[label_col], random_state=seed
    )
    return train, val, test


def report_imbalance(df: pd.DataFrame, label_col: str, name: str) -> None:
    counts = df[label_col].value_counts()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    ratio = neg / max(pos, 1)
    print(f"[{name}] n={len(df)} pos={pos} neg={neg} neg:pos ~= {ratio:.1f}:1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Only read the first N rows (for fast local iteration on large files like Avazu's 40M-row train set)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)["data"]
    label_col = cfg["label_col"]
    drop_cols = cfg.get("drop_cols", [])
    numeric_cols = cfg["numeric_cols"]
    categorical_cols = cfg["categorical_cols"]

    # pandas infers compression (e.g. .gz) from the file extension automatically
    df = pd.read_csv(args.raw_path, nrows=args.nrows)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if not numeric_cols and not categorical_cols:
        # Auto-infer column roles by dtype when not explicitly set in the config (excluding the label column)
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != label_col
        ]
        categorical_cols = [c for c in df.columns if c not in numeric_cols + [label_col]]

    df = clean_missing(df, numeric_cols, categorical_cols)
    df, vocab_sizes = encode_categorical(df, categorical_cols)

    train, val, test = split_data(
        df, label_col, cfg["val_size"], cfg["test_size"], cfg["random_seed"]
    )

    report_imbalance(train, label_col, "train")
    report_imbalance(val, label_col, "val")
    report_imbalance(test, label_col, "test")

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(args.out_dir, "val.parquet"), index=False)
    test.to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)

    meta = {
        "label_col": label_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "vocab_sizes": vocab_sizes,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Preprocessing complete, output written to {args.out_dir}")


if __name__ == "__main__":
    main()
