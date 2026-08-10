"""Static schematics of the FM and Transformer CTR architectures
(src/models/fm.py, src/models/transformer.py), for use in the report/slides.
Not data-driven - fixed diagrams, regenerate with:

    python -m src.plot_architecture --model fm --out reports/figures/fm_architecture.png
    python -m src.plot_architecture --model transformer --out reports/figures/transformer_architecture.png
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


def build_fm_figure():
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


def build_transformer_figure():
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=SURFACE)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    cx = 4.5

    # Input
    in_y, in_h = 6.7, 0.7
    _box(ax, (0.4, in_y), 8.2, in_h,
         "Input: 22 categorical fields  x = [x_1, ..., x_22]  (Avazu, no numeric features)",
         facecolor="#ffffff", fontsize=9)

    # Offset lookup
    off_y, off_h = 5.65, 0.6
    _box(ax, (3.05, off_y), 2.9, off_h,
         "x + field offsets\n(shared-table lookup)", facecolor="#ffffff", fontsize=9)
    _arrow(ax, (cx, in_y), (cx, off_y + off_h))

    # Field embedding + positional embedding + embed dropout
    emb_y, emb_h = 4.45, 0.95
    _box(ax, (1.0, emb_y), 7.0, emb_h,
         "Field Embedding: Embedding(99149, embed_dim=16)\n"
         "+ learnable Field Positional Embedding  ->  (B, 22, 16)\n"
         "+ Dropout(p)  [embed_dropout]",
         facecolor=BLUE, textcolor="#ffffff", fontsize=9)
    _arrow(ax, (cx, off_y), (cx, emb_y + emb_h))

    # Transformer encoder stack
    enc_y, enc_h = 2.9, 1.25
    _box(ax, (1.0, enc_y), 7.0, enc_h,
         "TransformerEncoder x num_layers (default 2)\n"
         "Multi-Head Self-Attention (default heads=4)\n"
         "+ FeedForward(dim=ff_dim=64) + residual + LayerNorm\n"
         "+ Dropout(p) on attn/FFN sublayers  ->  (B, 22, 16)",
         facecolor=ORANGE, textcolor="#ffffff", fontsize=9)
    _arrow(ax, (cx, emb_y), (cx, enc_y + enc_h))

    # Mean pooling
    pool_y, pool_h = 1.9, 0.6
    _box(ax, (2.2, pool_y), 4.6, pool_h,
         "Mean-Pool over 22 fields  ->  (B, embed_dim=16)",
         facecolor="#ffffff", fontsize=9)
    _arrow(ax, (cx, enc_y), (cx, pool_y + pool_h))

    # Head
    head_y, head_h = 0.9, 0.6
    _box(ax, (2.5, head_y), 4.0, head_h,
         "Dropout(p)  ->  Linear(embed_dim, 1)",
         facecolor=AQUA, textcolor="#ffffff", fontsize=9)
    _arrow(ax, (cx, pool_y), (cx, head_y + head_h))

    ax.text(4.5, 0.35, "raw logit -> BCEWithLogitsLoss / FocalLoss (sigmoid applied inside the loss)",
            ha="center", va="center", color=MUTED, fontsize=8.5, style="italic")
    ax.text(4.5, 0.05,
            "Note: flatten+MLP head was tried first and ruled out empirically (see IMPLEMENTATION_NOTES.md) -\n"
            "mean-pool removes per-field-position weights, forcing interactions into the shared attention layers.",
            ha="center", va="center", color=MUTED, fontsize=7.5, style="italic")

    ax.set_title("Transformer CTR (src/models/transformer.py)", color=INK, fontsize=13, pad=12)
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["fm", "transformer"], default="fm")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main():
    import os
    args = parse_args()
    out = args.out or f"reports/figures/{args.model}_architecture.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig = build_fm_figure() if args.model == "fm" else build_transformer_figure()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
