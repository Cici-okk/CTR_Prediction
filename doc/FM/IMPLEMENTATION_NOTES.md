# Implementation Notes: FM Baseline + Training/Optimization Pipeline

Detailed writeup of what was built to cover the "Methodology" and "Results & Optimization"
sections of the project. Assumes familiarity with the existing data pipeline
(`src/data/preprocess.py`, `src/data/dataset.py`, `src/losses/focal_loss.py`) — see the
main `README.md` for that part.

## 1. Model architecture — Factorization Machine (`src/models/fm.py`)

A standard 2-way FM over the 22 categorical fields (Avazu has no numeric features):

```
logit = bias
      + sum_i  w(x_i)                                              # order-1 (linear)
      + 0.5 * sum_k [ (sum_i v_i,k(x_i))^2 - sum_i v_i,k(x_i)^2 ]   # order-2 (pairwise interaction)
```

where `x_i` is the label-encoded category for field `i`, `w(x_i)` is a scalar per-value
weight, and `v_i(x_i)` is a `k`-dimensional embedding per value (`k = embed_dim`, default
16). The order-2 term is the standard FM trick for computing all pairwise interactions in
`O(fields * k)` instead of `O(fields^2 * k)`.

**Implementation detail — shared embedding table with offsets.** Rather than one
`nn.Embedding` per field (22 separate tables), the model concatenates all fields' vocabs
into a single table per order and looks values up by adding a per-field offset to the raw
index:

```python
field_dims = [vocab_sizes[c] for c in categorical_cols]   # e.g. [2, 6, 6, 1239, ...]
offsets = [0, field_dims[0], field_dims[0]+field_dims[1], ...]   # cumsum, registered as a buffer
x = categorical + offsets   # (B, 22) -> globally-unique indices into the shared table
```

Total vocab across all 22 fields is 99,149 (dominated by `device_ip` at 76,668 and
`device_id` at 15,040). At `embed_dim=16` the order-2 table is ~6.3 MB — trivial, so no
hashing/bucketing trick is needed here (that's more relevant to a high-cardinality
Transformer/DeepFM model, out of this scope). This mirrors the standard reference
implementation pattern used by libraries like `torchfm`.

**Initialization matters.** Both embedding tables are explicitly initialized
(`xavier_uniform_` for the order-2 table, zeros for the order-1/linear table and the
bias) rather than left at PyTorch's default `N(0,1)`. With 22 fields summed per sample,
default init produces huge, unstable initial logits — this is a correctness fix, not
cosmetic.

`forward(categorical)` returns raw logits of shape `(B,)`, pre-sigmoid — a drop-in match
for both `nn.BCEWithLogitsLoss` and the existing `FocalLoss`.

## 2. Loss: BCE vs Focal (config-driven)

`FocalLoss` (`src/losses/focal_loss.py`) already existed; `src/train.py` just makes the
choice a config/CLI switch instead of hardcoding one:

```python
criterion = nn.BCEWithLogitsLoss() if cfg["loss"]["type"] == "bce" \
    else FocalLoss(gamma=cfg["loss"]["focal_gamma"], alpha=cfg["loss"]["focal_alpha"])
```
`--loss bce|focal` overrides `configs/config.yaml`'s `loss.type` from the CLI.

## 3. Optimizer: Adam vs AdamW

Both are built with the **same** `lr` and `weight_decay` from config — the entire point
of the comparison is isolating the optimizer's internal behavior, not tuning separate
hyperparameters per optimizer:

```python
torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

**Why they differ**: `Adam`'s `weight_decay` is implemented as classic L2 regularization
— it's added directly to the gradient before the adaptive (per-parameter, second-moment
normalized) update, so its effective strength gets rescaled by each parameter's own
adaptive learning rate. Parameters with small gradient history (e.g. rare category
embeddings, which see gradients infrequently) end up **under-regularized** relative to
what `weight_decay` nominally specifies. `AdamW` decouples weight decay from the gradient
step entirely — it's applied directly to the weights (`w -= lr * weight_decay * w`)
outside of the adaptive-moment machinery, so every parameter gets the same proportional
decay regardless of its gradient history. This matters here specifically because of the
extreme sparsity gradient: `device_ip`/`device_id`/`site_id` embeddings have huge vocabs
where most values are rare, so plenty of parameters get very few updates — the exact
scenario where Adam's coupled decay under-regularizes.

`--optimizer adam|adamw` overrides `train.optimizer`.

## 4. Regularization: Dropout + Early Stopping + LR Warmup

- **Dropout** (`model.dropout`, default 0.2): applied to the `(B, 22, embed_dim)` field
  embeddings before the order-2 interaction sum, so it can zero out individual field
  embeddings per sample per step — the standard place to regularize an FM/DeepFM-style
  interaction term. `--dropout <float>` overrides it (`--dropout 0.0` disables it).

- **Early Stopping** (`src/utils.py::EarlyStopping`): monitors **validation AUC** (mode
  `max`) by default, with `patience` from `train.early_stopping_patience` (3). Design
  choice: **best-value tracking always runs, independent of whether stopping is
  enabled.** This matters for the ablation runs — a `--no_early_stop` run still needs
  its best-epoch checkpoint saved for a fair final test-set number, it just never breaks
  the training loop early. `enabled=False` is a distinct flag from "patience=infinity",
  not a hack.

  **Why AUC and not LogLoss as the monitored metric**: Focal Loss's `(1-p_t)^γ` term
  deliberately distorts probability calibration relative to plain BCE (that's the whole
  point of down-weighting easy examples). LogLoss is sensitive to calibration, so using
  it as the stopping/selection criterion would unfairly bias the BCE-vs-Focal comparison
  toward whichever loss happens to produce better-calibrated (rather than
  better-ranking) probabilities. AUC is invariant to any monotonic recalibration, so it's
  the fair common yardstick — and it directly reflects what CTR ad-ranking actually
  optimizes for. LogLoss is still computed and logged every epoch, just not the trigger.
  Configurable via `train.early_stopping_metric: auc | logloss`.

- **LR Warmup** (`train.warmup_steps`, default 500): linear warmup implemented via
  `torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step+1)/warmup_steps))`,
  stepped once per **batch** (not per epoch) — so with `batch_size=1024` and 160,000
  training rows, warmup covers roughly the first 3.2 epochs before holding flat at the
  configured `lr`.

## 5. Metrics: AUC and LogLoss (`src/utils.py::compute_metrics`)

Two correctness details that are easy to get wrong here:

1. **AUC and LogLoss are not batch-averageable.** Both are computed once per split per
   epoch over the full concatenated set of predictions/labels (`roc_auc_score` is a
   global ranking statistic; averaging per-batch AUCs is a different, wrong number).
2. **LogLoss needs probabilities, not logits.** `sigmoid(logits)` is computed once and
   fed to both `roc_auc_score` (rank-invariant to the transform, so it wouldn't matter
   for AUC) and `log_loss` (which absolutely requires values in `[0, 1]` — feeding it raw
   logits silently produces a nonsense loss after sklearn's internal clipping).

## 6. Train/Val gap diagnosis — and why there are *three* train-related dataloaders

`src/train.py` builds:
- `train_loader` — wrapped in a `WeightedRandomSampler` when `sampling.strategy ==
  class_weight` (the default), used **only** for gradient updates.
- `train_eval_loader` — the same `train.parquet`, but `strategy="none", shuffle=False`
  (plain, unsampled), used **only** to compute the per-epoch train-side AUC/LogLoss.
- `val_loader` / `test_loader` — plain, unsampled, as expected.

**Why the separate `train_eval_loader` is required, not a nice-to-have**: if the
per-epoch "train metric" were computed by reusing the sampler-wrapped `train_loader`
(or reusing predictions made mid-epoch during the training pass), it would be measured
on an artificially class-rebalanced distribution (oversampled positives) and/or with
dropout active and stale weights. Neither is comparable to `val`'s natural ~17.4%
positive rate, which would corrupt exactly the train/val-gap number this whole exercise
exists to produce. Every epoch, after the training pass, the model is set to `.eval()`
and run once cleanly over both `train_eval_loader` and `val_loader` before computing
metrics — the extra forward-only passes are cheap (no backward pass, ~157+20 batches on
an FM model) next to the training pass itself.

`gap = train_auc - val_auc` is recorded every epoch in `history.json`.

## 7. Results from the required ablations

All five runs used the locked 160k/20k/20k Avazu split, `embed_dim=16`, `lr=1e-3`,
`weight_decay=1e-5`, `warmup_steps=500`, `batch_size=1024`. Full per-epoch data is in
each run's `history.json`; summarized final numbers (best epoch selected by val AUC):

| run | loss | optimizer | dropout | early stop | best epoch | val AUC | val LogLoss | test AUC | test LogLoss |
|---|---|---|---|---|---|---|---|---|---|
| `bce_adamw` | bce | adamw | 0.2 | on | 4 | 0.7625 | 0.5123 | 0.7706 | 0.5066 |
| `focal_adamw` | focal | adamw | 0.2 | on | 4 | 0.7624 | 0.4804 | 0.7718 | 0.4773 |
| `focal_adam` | focal | adam | 0.2 | on | 4 | 0.7642 | 0.4919 | 0.7731 | 0.4896 |
| `with_reg` | focal | adamw | 0.2 | on | 4 | 0.7624 | 0.4804 | 0.7718 | 0.4773 |
| `no_reg` (full 20 epochs) | focal | adamw | 0.0 | **off** | 2 | 0.7615 | 0.5009 | 0.7723 | 0.4983 |

**BCE vs Focal**: AUC is essentially tied (~0.762 val), but Focal gives a meaningfully
lower LogLoss (0.480 vs 0.512 val) — consistent with Focal's down-weighting of easy
negatives producing better-calibrated probabilities on this ~1:4.7 imbalance, without
sacrificing ranking quality.

**Adam vs AdamW**: at this scale/duration the difference is small (val AUC 0.7642 vs
0.7624), within run-to-run noise — the theoretical decoupling advantage of AdamW is
real (see §3) but its practical effect here is modest given only 5 epochs of training
before early stopping triggers; it would be expected to matter more with heavier
weight_decay or longer training on the long-tail embeddings.

**Overfitting diagnosis (dropout + early stopping)**: this is the clearest result. The
`no_reg` run (dropout=0, early stopping disabled, forced through all 20 epochs) shows
its best val AUC at **epoch 2**, then keeps degrading while train AUC keeps climbing —
by epoch 19, train AUC reaches 0.9706 against val AUC 0.7276, a gap of **0.243**, more
than 4x the gap at its own best epoch (0.0973). The `with_reg` run (dropout=0.2, early
stopping on) tracks a visibly slower-growing gap and **stops itself at epoch 7** once
val AUC has plateaued/declined for 3 consecutive epochs past its best (epoch 4, gap
0.166) — avoiding 12+ epochs of pure overfitting the `no_reg` run wastes compute on.
See `runs/comparisons/with-reg_vs_no-reg_gap.png` for the plot.

## 8. Known gotcha fixed along the way

`configs/config.yaml` originally had `lr: 1e-3` / `weight_decay: 1e-5`. PyYAML's
`safe_load` (YAML 1.1 spec) only recognizes scientific notation as a float when it has
an explicit decimal point or sign in the mantissa — `1e-3` without a `.0` parses as a
**string**, not `0.001`. This silently broke `torch.optim.AdamW(..., lr="1e-3")` (a
`TypeError` inside `Adam.__init__`'s `0.0 <= lr` check). Fixed by writing `1.0e-3` /
`1.0e-5` in the yaml, plus a defensive `float(...)` cast in `train.py::build_optimizer`
so this class of bug can't resurface silently if someone edits the yaml back to the
ambiguous form.

## 9. File map

| File | Status | Purpose |
|---|---|---|
| `src/models/fm.py` | new | `FactorizationMachine` model |
| `src/train.py` | new | training loop, CLI, checkpointing, metrics/history logging |
| `src/plot_results.py` | new | run comparison table + train/val curve + gap plots |
| `src/utils.py` | modified | added `compute_metrics`, `EarlyStopping` (kept `set_seed`) |
| `configs/config.yaml` | modified | added `model:` section, `train.early_stopping_metric`, fixed `lr`/`weight_decay` float formatting |
