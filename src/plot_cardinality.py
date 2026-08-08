"""Evidence chart for the Transformer overfitting diagnosis (see
doc/Transformer/IMPLEMENTATION_NOTES.md): for each high-cardinality categorical
field, what fraction of training rows have a value that appears exactly once in
the training set? A "singleton" value is effectively a row ID rather than a
generalizable category - self-attention's query/key routing can exploit it to
memorize individual rows, which FM's closed-form pairwise interaction cannot.

Usage:
    python -m src.plot_cardinality --out reports/figures/field_cardinality.png
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BAR_COLOR = "#eb6834"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, zorder=0, axis="y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def compute_singleton_fractions(processed_dir: str, top_k: int) -> pd.DataFrame:
    with open(os.path.join(processed_dir, "meta.json")) as f:
        meta = json.load(f)

    fields = sorted(meta["categorical_cols"], key=lambda c: meta["vocab_sizes"][c], reverse=True)[:top_k]
    df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"), columns=fields)

    rows = []
    for field in fields:
        counts = df[field].value_counts()
        rows_with_singleton = df[field].map(counts).eq(1).mean()
        rows.append({
            "field": field,
            "vocab_size": meta["vocab_sizes"][field],
            "unique_in_train": len(counts),
            "singleton_id_frac": (counts == 1).mean(),
            "rows_with_singleton_frac": rows_with_singleton,
        })
    return pd.DataFrame(rows).sort_values("rows_with_singleton_frac", ascending=False).reset_index(drop=True)


def plot_singleton_fractions(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    x = range(len(df))
    values = df["rows_with_singleton_frac"].values * 100
    bars = ax.bar(x, values, color=BAR_COLOR, width=0.6, zorder=2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}%",
                ha="center", va="bottom", color=INK, fontsize=9)

    labels = [f"{f}\n({v:,} values)" for f, v in zip(df["field"], df["vocab_size"])]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("% of training rows with a singleton value")
    ax.set_title("Near-unique categorical IDs in training data (Avazu)", color=INK)
    ax.set_ylim(0, max(values) * 1.25)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--out", type=str, default="reports/figures/field_cardinality.png")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = compute_singleton_fractions(args.processed_dir, args.top_k)
    print(df.to_string(index=False))
    plot_singleton_fractions(df, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
