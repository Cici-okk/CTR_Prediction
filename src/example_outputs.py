"""Generates the report's required "Example Outputs" section: one concrete test-set
row for each of the four FM/Transformer agreement-vs-disagreement outcomes (both
correct, FM-only correct, Transformer-only correct, both wrong), selected by a fixed,
non-arbitrary rule - the first test-set row (in natural index order) matching each
outcome - so there is no room to cherry-pick a flattering example.

(For the aggregate, non-anecdotal comparison used elsewhere in Results, see
src/inspect_predictions.py's memorization-susceptible-vs-common subgroup analysis.)

Usage:
    python -m src.example_outputs \
        --fm_run runs/focal_adam --transformer_run runs/transformer_embeddrop
"""
import argparse
import json
import os

import pandas as pd
import torch
import yaml

from src.models.fm import FactorizationMachine
from src.models.transformer import TransformerCTR


def load_model(run_dir: str, meta: dict):
    with open(os.path.join(run_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_type = cfg["model"].get("type", "fm")
    if model_type == "transformer":
        model = TransformerCTR(
            meta["vocab_sizes"], meta["categorical_cols"],
            embed_dim=cfg["model"]["embed_dim"], dropout=cfg["model"]["dropout"],
            **cfg["model"].get("transformer", {}),
        )
    else:
        model = FactorizationMachine(
            meta["vocab_sizes"], meta["categorical_cols"],
            embed_dim=cfg["model"]["embed_dim"], dropout=cfg["model"]["dropout"],
        )
    state = torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--fm_run", type=str, default="runs/focal_adam")
    parser.add_argument("--transformer_run", type=str, default="runs/transformer_embeddrop")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.processed_dir, "meta.json")) as f:
        meta = json.load(f)

    fm = load_model(args.fm_run, meta)
    transformer = load_model(args.transformer_run, meta)

    test_df = pd.read_parquet(os.path.join(args.processed_dir, "test.parquet"))
    categorical = torch.tensor(test_df[meta["categorical_cols"]].values, dtype=torch.long)
    with torch.no_grad():
        fm_probs = torch.sigmoid(fm(categorical)).numpy()
        tr_probs = torch.sigmoid(transformer(categorical)).numpy()
    labels = test_df[meta["label_col"]].values

    fm_correct = (fm_probs >= args.threshold).astype(int) == labels
    tr_correct = (tr_probs >= args.threshold).astype(int) == labels

    outcomes = [
        ("Both correct", fm_correct & tr_correct),
        ("FM correct, Transformer wrong", fm_correct & ~tr_correct),
        ("Transformer correct, FM wrong", ~fm_correct & tr_correct),
        ("Both wrong", ~fm_correct & ~tr_correct),
    ]

    for name, mask in outcomes:
        print(f"=== {name} (n={mask.sum()} in test set) ===")
        if not mask.any():
            print("  (no test-set row matches this outcome)")
            print()
            continue
        idx = mask.argmax()  # first True index in natural order
        row = test_df.iloc[idx]
        print(f"  Test row index: {idx}")
        print(f"  Ground truth label (click): {int(row[meta['label_col']])}")
        print(f"  FM          -> prob={fm_probs[idx]:.4f}  (pred={'1' if fm_probs[idx] >= args.threshold else '0'})")
        print(f"  Transformer -> prob={tr_probs[idx]:.4f}  (pred={'1' if tr_probs[idx] >= args.threshold else '0'})")
        print(f"  device_ip={row['device_ip']}, device_id={row['device_id']}, site_id={row['site_id']}")
        print()


if __name__ == "__main__":
    main()
