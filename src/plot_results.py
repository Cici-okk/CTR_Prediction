"""Compare training runs produced by src/train.py: a summary table plus
train/val curve and train-val-gap plots, for the BCE-vs-Focal,
Adam-vs-AdamW, and dropout+early-stopping-vs-none ablations.

Usage:
    python -m src.plot_results --runs runs/with_reg runs/no_reg \
        --labels with-reg no-reg --out_dir runs/comparisons
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

# Categorical run colors (fixed order, from the project's validated palette;
# see the dataviz skill's references/palette.md). Split (train/val) is
# encoded by linestyle, not color, so the two together never blow past a
# 3-run categorical budget. A 4th slot (yellow) is added for plot_overview,
# which compares up to 4 runs as adjacent bars with mandatory direct labels.
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "history.json")) as f:
        history = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    return history, metrics


def comparison_table(run_dirs: list, labels: list) -> pd.DataFrame:
    rows = []
    for run_dir, label in zip(run_dirs, labels):
        _, metrics = load_run(run_dir)
        rows.append({
            "run": label,
            "loss_type": metrics["loss_type"],
            "optimizer": metrics["optimizer"],
            "dropout": metrics["dropout"],
            "best_epoch": metrics["best_epoch"],
            "val_auc": metrics["val_auc"],
            "val_logloss": metrics["val_logloss"],
            "test_auc": metrics["test_auc"],
            "test_logloss": metrics["test_logloss"],
        })
    return pd.DataFrame(rows)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_curves(run_dirs: list, labels: list, metric: str, out_path: str):
    """metric: 'auc' or 'logloss'. One axis only: never overlays both metrics."""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        history, _ = load_run(run_dir)
        df = pd.DataFrame(history)
        color = RUN_COLORS[i % len(RUN_COLORS)]
        ax.plot(df["epoch"], df[f"train_{metric}"], color=color, linestyle="-",
                linewidth=2, marker="o", markersize=5, label=f"{label} (train)")
        ax.plot(df["epoch"], df[f"val_{metric}"], color=color, linestyle="--",
                linewidth=2, marker="o", markersize=5, label=f"{label} (val)")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric.upper() if metric == "auc" else "LogLoss")
    ax.set_title(f"Train vs Val {metric.upper() if metric == 'auc' else 'LogLoss'}", color=INK)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap(run_dirs: list, labels: list, out_path: str):
    """train_auc - val_auc per epoch per run: the overfitting diagnosis chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="-", zorder=1)
    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        history, _ = load_run(run_dir)
        df = pd.DataFrame(history)
        color = RUN_COLORS[i % len(RUN_COLORS)]
        ax.plot(df["epoch"], df["gap"], color=color, linewidth=2,
                marker="o", markersize=5, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("train AUC - val AUC")
    ax.set_title("Train/Val AUC Gap (overfitting diagnosis)", color=INK)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview(run_dirs: list, labels: list, out_path: str):
    """Final test AUC and test LogLoss across all runs, side by side (small
    multiples, each its own single-axis panel - never a shared dual axis).
    Bars start at true zero (no truncated axis); exact values are direct-
    labeled at the bar cap since the differences between runs are small."""
    metrics_list = [load_run(rd)[1] for rd in run_dirs]
    colors = [RUN_COLORS[i % len(RUN_COLORS)] for i in range(len(labels))]
    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=SURFACE)

    for ax, key, title in (
        (axes[0], "test_auc", "Test AUC (higher is better)"),
        (axes[1], "test_logloss", "Test LogLoss (lower is better)"),
    ):
        values = [m[key] for m in metrics_list]
        bars = ax.bar(x, values, color=colors, width=0.6, zorder=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title, color=INK)
        ax.set_ylim(0, max(values) * 1.15)
        _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default="runs/comparisons")
    parser.add_argument("--overview", action="store_true",
                         help="Also emit a bar-chart overview of final test AUC/LogLoss across all --runs")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = args.labels if args.labels else [os.path.basename(r.rstrip("/")) for r in args.runs]
    os.makedirs(args.out_dir, exist_ok=True)

    tag = "_vs_".join(labels)

    table = comparison_table(args.runs, labels)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(args.out_dir, f"{tag}_table.csv"), index=False)

    plot_curves(args.runs, labels, "auc", os.path.join(args.out_dir, f"{tag}_auc.png"))
    plot_curves(args.runs, labels, "logloss", os.path.join(args.out_dir, f"{tag}_logloss.png"))
    plot_gap(args.runs, labels, os.path.join(args.out_dir, f"{tag}_gap.png"))

    if args.overview:
        plot_overview(args.runs, labels, os.path.join(args.out_dir, f"{tag}_overview.png"))


if __name__ == "__main__":
    main()
