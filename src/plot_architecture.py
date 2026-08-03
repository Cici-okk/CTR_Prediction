"""Static schematic of the FactorizationMachine architecture (src/models/fm.py),
for use in the report/slides. Not data-driven - a fixed diagram, regenerate with:

    python -m src.plot_architecture --out reports/figures/fm_architecture.png
"""
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def _box(ax, xy, w, h, text, facecolor, textcolor=INK, fontsize=10):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor=MUTED, facecolor=facecolor, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=textcolor, fontsize=fontsize, zorder=3, wrap=True)
    return (x + w / 2, y), (x + w / 2, y + h)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=MUTED, zorder=1,
    ))


def build_figure():
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # Input
    top_in, bot_in = _box(ax, (0.4, 5.0), 8.2, 0.7,
                           "Input: 22 categorical fields  x = [x_1, ..., x_22]  (Avazu, no numeric features)",
                           facecolor="#ffffff", fontsize=9)

    # Offset lookup
    top_off, bot_off = _box(ax, (3.05, 3.95), 2.9, 0.6,
                             "x + field offsets\n(shared-table lookup)", facecolor="#ffffff", fontsize=9)
    _arrow(ax, (4.5, top_in[1] - 0.7), (4.5, bot_off[1]))

    # Two embedding tables
    top_lin, bot_lin = _box(ax, (0.6, 2.6), 3.4, 0.9,
                             "Order-1 (linear)\nEmbedding(99149, 1)\n+ bias", facecolor=BLUE, textcolor="#ffffff")
    top_v, bot_v = _box(ax, (5.0, 2.6), 3.4, 0.9,
                         "Order-2 (interaction)\nEmbedding(99149, embed_dim=16)\n+ Dropout(p)", facecolor=ORANGE, textcolor="#ffffff")
    _arrow(ax, (4.0, bot_off[1] - 0.4), (2.3, top_lin[1]))
    _arrow(ax, (5.0, bot_off[1] - 0.4), (6.7, top_v[1]))

    # Sum / FM trick
    top_sum1, bot_sum1 = _box(ax, (0.6, 1.4), 3.4, 0.7,
                               "sum_i w(x_i) + bias", facecolor="#ffffff", fontsize=9)
    top_sum2, bot_sum2 = _box(ax, (5.0, 1.4), 3.4, 0.9,
                               "0.5 * sum_k[ (sum_i v_i)^2\n- sum_i v_i^2 ]", facecolor="#ffffff", fontsize=9)
    _arrow(ax, (2.3, bot_lin[1]), (2.3, top_sum1[1] + 0.7))
    _arrow(ax, (6.7, bot_v[1]), (6.7, top_sum2[1] + 0.9))

    # Combine
    top_c, bot_c = _box(ax, (3.05, 0.3), 2.9, 0.6, "logit = order-1 + order-2", facecolor=AQUA, textcolor="#ffffff", fontsize=9)
    _arrow(ax, (2.3, bot_sum1[1]), (3.8, top_c[1] + 0.6))
    _arrow(ax, (6.7, bot_sum2[1]), (5.2, top_c[1] + 0.6))

    ax.text(4.5, 0.05, "raw logit -> BCEWithLogitsLoss / FocalLoss (sigmoid applied inside the loss)",
            ha="center", va="center", color=MUTED, fontsize=8.5, style="italic")

    ax.set_title("Factorization Machine (src/models/fm.py)", color=INK, fontsize=13, pad=12)
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="reports/figures/fm_architecture.png")
    return parser.parse_args()


def main():
    import os
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig = build_figure()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
