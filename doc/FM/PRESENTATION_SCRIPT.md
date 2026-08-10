# Presentation Script — Methodology + Results & Optimization

Two spoken-word scripts for presenting this section of the project. Figures referenced
are in `reports/figures/`. Timings assume ~150 words/minute.

---

## 3-minute version (~440 words)

**[Show: `fm_architecture.png`]**

Our baseline model for CTR prediction is a Factorization Machine, or FM. The Avazu
dataset gives us 22 categorical features — things like site ID, device ID, hour of day
— and no numeric features at all. FM handles this by giving every category value two
embeddings: a 1-dimensional one for a linear, first-order effect, and a k-dimensional
one, we used 16, for second-order effects. That second part is the real value of FM: it
learns pairwise interactions between every pair of features — like "this device model on
this site category" — without ever explicitly building a device-model-by-site-category
table, which would be far too large. Both embedding tables share one lookup, indexed by
per-field offsets, since our vocab sizes range from just 2 for hour up to over 76,000
distinct values for device IP.

**[Show: `loss_comparison_logloss.png`]**

For the loss function, we compared standard BCE against Focal Loss. Our label is
imbalanced, about 1 click for every 4.7 non-clicks. Focal Loss down-weights the easy,
already-confident negatives and lets the model focus on harder examples. In our
experiments the two losses land at almost the same AUC, but Focal gives a meaningfully
lower validation LogLoss — 0.48 versus 0.51 for BCE — meaning better-calibrated click
probabilities, which matters a lot if these scores feed into real ad-auction bidding.

For optimization, we compared Adam against AdamW with identical learning rate and weight
decay. AdamW decouples weight decay from the gradient update, so every embedding gets
consistent regularization regardless of how often it's updated — which matters here
because features like device ID have a long tail of rarely-seen values. We layered in
Dropout on the embeddings, Early Stopping monitored on validation AUC, and a linear
learning-rate warmup over the first 500 steps.

**[Show: `overfitting_gap.png`]**

The clearest result is our overfitting diagnosis. We tracked the train-minus-validation
AUC gap every epoch. Without dropout and without early stopping, forced through all 20
epochs, that gap grows to 0.24 — the model is essentially memorizing training rows. With
dropout and early stopping turned on, training halts itself at epoch 7, right after
validation AUC plateaus, with a gap of just 0.17 at its best epoch — and it saves the
compute of the remaining 12+ wasted epochs.

**[Show: `results_overview.png`]**

Final numbers: all four configurations land around 0.77 test AUC — tightly clustered,
as expected for an FM baseline — but Focal Loss consistently wins on LogLoss, and
Dropout plus Early Stopping is what keeps the model honest instead of overfit.

---

## 1-minute version (~145 words)

We built a Factorization Machine baseline for CTR prediction over Avazu's 22 categorical
features — no numeric features here — modeling both individual feature effects and
pairwise interactions between features, like device-model-by-site, without an explosion
in parameters.

**[Show: `loss_comparison_logloss.png`]** We compared standard BCE against Focal Loss for
this imbalanced 1-in-4.7 click rate: AUC ties, but Focal gives a meaningfully lower
LogLoss — better-calibrated probabilities.

**[Show: `overfitting_gap.png`]** For optimization we compared Adam versus AdamW, and
combined Dropout, Early Stopping, and LR warmup. The overfitting story is the clearest
result: without regularization the train/val AUC gap balloons to 0.24 over 20 epochs;
with Dropout and Early Stopping, training self-stops at epoch 7 with a gap of just 0.17
— avoiding wasted overfitting.

**[Show: `results_overview.png`]** All configurations land around 0.77 test AUC, with
Focal Loss winning on LogLoss.
