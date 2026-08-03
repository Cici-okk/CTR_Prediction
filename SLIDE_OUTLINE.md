# 1-Page Slide Outline — Methodology + Results & Optimization

Layout target: one slide, 3 columns (Method / Experiment / Result) or a 2x2 grid with
figures. Figures are in `reports/figures/`.

---

**Title:** CTR Prediction — FM Baseline: Loss, Optimizer & Overfitting Study

**Column 1 — Methodology**
- Model: Factorization Machine (FM) over 22 categorical fields (Avazu, no numeric
  features) — order-1 linear term + order-2 pairwise-interaction term, embed_dim=16
  - *Figure: `fm_architecture.png`*
- Loss: standard BCE vs Focal Loss (γ=2.0, α=0.25) — Focal down-weights easy negatives
  under the ~1:4.7 class imbalance

**Column 2 — Experiment**
- Optimizer: Adam vs AdamW (identical lr=1e-3, weight_decay=1e-5) — AdamW decouples
  weight decay from the adaptive update, better for long-tail rare embeddings
- Regularization stack: Dropout (0.2) + Early Stopping (on val AUC, patience 3) +
  Linear LR Warmup (500 steps)
- Overfitting diagnosis: track train−val AUC gap every epoch, with vs without
  Dropout+Early Stopping
  - *Figure: `overfitting_gap.png`*

**Column 3 — Results**
- BCE vs Focal: AUC ties (~0.762 val); **Focal wins LogLoss** (0.480 vs 0.512 val) →
  better-calibrated probabilities
- Adam vs AdamW: AUC 0.764 vs 0.762 (small at this scale, expected to widen with more
  training / stronger decay)
- **No regularization**: train/val AUC gap grows to **0.24** over 20 epochs, val AUC
  peaks early (epoch 2) then degrades
- **With Dropout + Early Stopping**: self-stops at epoch 7, gap only **0.17** at its
  best epoch (epoch 4) — avoids 12+ epochs of wasted overfitting
  - *Figure: `results_overview.png`* (all configs land ~0.77 test AUC)

**Footer / takeaway:** Focal Loss improves calibration without sacrificing ranking
quality; Dropout + Early Stopping is what actually controls overfitting on this data.

---

## Speaker note
Pair with `PRESENTATION_SCRIPT.md` — use the 1-minute version if this is a single slide
in a multi-team-member deck, the 3-minute version if this section gets its own slot.
