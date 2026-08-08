# Implementation Notes: Transformer CTR Model + Overfitting Diagnosis

Detailed writeup of what was built and diagnosed to extend the FM baseline (see
`doc/FM/IMPLEMENTATION_NOTES.md`) with a Transformer/self-attention model over the same
22 categorical fields. Assumes familiarity with that document and the data pipeline it
describes.

## 1. Model architecture — Transformer CTR (`src/models/transformer.py`)

AutoInt-style feature interaction: each field's embedding is treated as one token in a
length-22 sequence, and multi-head self-attention over that sequence learns feature
interactions automatically, in place of FM's hand-derived closed-form second-order term.

```
tokens  = dropout(field_embedding(x) + field_positional_embedding)   # (B, 22, d)
tokens  = TransformerEncoder(tokens)                                  # (B, 22, d)
logit   = Linear(dropout(mean_pool(tokens)))                          # (B,)
```

- **Shared embedding table with offsets** — identical mechanism to FM's (`src/models/fm.py`):
  all 22 fields' vocabs are concatenated into one `nn.Embedding(99149, embed_dim)` and
  looked up via a per-field offset buffer, so `TransformerCTR.__init__` takes the exact
  same required signature as `FactorizationMachine` (`vocab_sizes, categorical_cols,
  embed_dim=16, dropout=0.2`), plus optional `num_heads=4, num_layers=2, ff_dim=64`. This
  makes it a drop-in swap in `src/train.py` (see §3).
- **Field positional embedding** — a learnable `(1, 22, embed_dim)` parameter added to
  every token. Necessary because self-attention is permutation-invariant, but the 22
  fields are not interchangeable (`hour` and `device_id` have distinct semantics/vocab);
  without it the model can't tell *which* field an embedding came from.
- **`nn.TransformerEncoder`** — standard multi-head self-attention + FFN sublayers,
  `dropout` shared across attention/FFN internals.

`forward(categorical)` returns raw logits of shape `(B,)`, pre-sigmoid — same contract
as FM, works with both `nn.BCEWithLogitsLoss` and `FocalLoss` unchanged.

## 2. Head ablation: Flatten+MLP → Mean-pool (empirical, not just theoretical)

The head was built and changed twice, based on actual run results rather than a priori
design:

1. **First attempt — Flatten+MLP**: `Flatten -> Linear(22*16, 64) -> ReLU -> Dropout ->
   Linear(64, 1)`. Hypothesis going in: this is the highest-capacity, most expressive
   option (AutoInt's paper uses a similar flatten+dense readout).
2. **Observed problem**: `runs/transformer_run` showed train AUC jumping from 0.769 to
   0.955 within 4 epochs while val AUC *fell* from 0.727 to 0.718 over the same span —
   train/val gap grew to **0.236** by epoch 4, several times larger than FM's gap at its
   own best epoch (~0.17, see FM notes §7). Suspected cause: flattening gives the head a
   *separate weight block per field position*, letting it memorize specific field-value
   combinations instead of learning generalizable interactions.
3. **Second attempt — Mean-pool + single Linear**: `mean(dim=1) -> Dropout ->
   Linear(embed_dim, 1)`, removing the per-position weight entirely.
4. **Result — hypothesis rejected**: `runs/transformer_meanpool` gap at epoch 4 was
   **0.246**, if anything slightly *worse* than Flatten+MLP's 0.236, and test AUC was
   statistically unchanged (0.7387 vs 0.7398). Head capacity was **not** the driver of
   the overfitting. Mean-pool was kept anyway (it's simpler and not worse), but the real
   cause had to be somewhere else — see §4.

This negative result is reported deliberately: it eliminated an entire hypothesis
class (head capacity) and redirected the investigation toward the embedding table
itself, which turned out to be the right track (§5).

## 3. Integration into the training pipeline (`src/train.py`)

```python
def build_model(meta: dict, cfg: dict) -> nn.Module:
    model_type = cfg["model"].get("type", "fm")
    if model_type == "transformer":
        return TransformerCTR(
            meta["vocab_sizes"], meta["categorical_cols"],
            embed_dim=cfg["model"]["embed_dim"], dropout=cfg["model"]["dropout"],
            **cfg["model"].get("transformer", {}),
        )
    return FactorizationMachine(...)
```

`--model_type fm|transformer` overrides `configs/config.yaml`'s `model.type` from the
CLI, mirroring the existing `--loss`/`--optimizer`/`--dropout` pattern. Nothing else in
the training loop, checkpointing, or evaluation changes — both models share the same
`forward(categorical) -> (B,) logits` contract.

## 4. Overfitting diagnosis, step by step

With the mean-pool head, `best_epoch` was still stuck at epoch 1 in every run, with
train AUC reaching the 0.90s by epoch 2-3 while val AUC declined from there. Three
regularization levers were tested in sequence, each with a clear pass/fail read before
moving to the next:

**Step 1 — missing embedding dropout (real parity gap, fixed).** FM applies
`dropout(embedding(x))` directly to the `(B, 22, embed_dim)` field embeddings before the
interaction sum. The Transformer's tokens went straight into the encoder with no dropout
of their own (only the encoder's *internal* attention/FFN dropout existed). Added
`self.embed_dropout` before the encoder. Effect: `runs/transformer_embeddrop` improved
test AUC from 0.7387 to **0.7477** — a real, if modest, gain. Kept.

**Step 2 — `embedding_weight_decay` sweep (ruled out, with an explanation).** Added a
param-group split in `build_optimizer` so the embedding table can carry its own
`weight_decay`, independent of the rest of the model:

```python
for name, param in model.named_parameters():
    (embedding_params if name.startswith("embedding.") else other_params).append(param)
param_groups = [
    {"params": embedding_params, "weight_decay": embedding_weight_decay},
    {"params": other_params, "weight_decay": weight_decay},
]
```

Tested `embedding_weight_decay` at 1e-4, 1e-1, and 1.0 (three orders of magnitude) —
**all three produced virtually identical results** (test AUC 0.7477 / 0.7477 / 0.7476,
epoch-by-epoch numbers matching to 3-4 decimal places). This was not "too small a dose,"
it was a structural dead end: `nn.TransformerEncoderLayer` applies **LayerNorm**
immediately after the residual add in each sublayer, and the Q/K/V projection matrices
sitting between the embedding table and that LayerNorm are themselves trainable and
*not* subject to the extra decay. Any magnitude shrinkage weight decay imposes on the
embedding table can be, and empirically is, compensated for by those downstream
projections rescaling during training — weight decay constrains vector *magnitude*, but
the discriminative signal being exploited (see §5) lives in vector *direction*, which
magnitude decay cannot touch. No further weight_decay values were tried; the mechanism,
not the dosage, was the problem.

**Step 3 — blanket `dropout=0.4` (partial signal, wrong tool).** `runs/transformer_embwd_1e1_do4`
raised the single shared `dropout` value (which feeds embed/attention/FFN/head
simultaneously) from 0.2 to 0.4. This *did* change behavior — epoch-1 train AUC dropped
from ~0.90-0.92 to **0.778**, and gap at epoch 1 fell from ~0.16-0.19 to **0.047** —
confirming that randomly zeroing activations (dropout), unlike shrinking weight
magnitude (decay), can interfere with the memorization mechanism. But applying it
uniformly to all four sublayers cost too much useful capacity: by epoch 4 the gap had
still climbed back to 0.252 (barely better than baseline), and final test performance
got worse, not better (test AUC 0.7431, test LogLoss **0.5316** vs 0.4664 without the
change) — the model underfit before it stopped overfitting. Right direction, wrong
granularity (a separate, higher embed-only dropout — decoupled from encoder/head dropout
— was identified as the natural next experiment, not yet implemented).

## 5. Root cause: near-unique categorical IDs

`reports/figures/field_cardinality.png` (generated by `src/plot_cardinality.py`) plots,
for each high-cardinality field, the fraction of the 160,000 training rows whose value
in that field is a **singleton** — appears exactly once in the entire training set:

| field | vocab size | unique values in train | % of training rows with a singleton value |
|---|---|---|---|
| `device_ip` | 76,668 | 66,560 | **26.9%** |
| `device_id` | 15,040 | 12,659 | 6.1% |
| `device_model` | 3,104 | 2,967 | 0.4% |
| `site_domain` | 1,119 | 1,032 | 0.2% |
| `site_id` | 1,239 | 1,167 | 0.2% |
| `app_id` | 989 | 936 | 0.2% |

For over a quarter of all training rows, `device_ip` is effectively a **row ID**, not a
generalizable category. Self-attention's query/key mechanism is structurally suited to
exploiting this: it can learn to use a field's embedding as a key that near-uniquely
identifies a training row, then route the label associated with that row directly to the
output — a discriminative shortcut, not a learned interaction. FM's closed-form pairwise
term (`0.5 * sum_k[(sum_i v_i)^2 - sum_i v_i^2]`) has no mechanism to express "if this
exact vector then this label"; it is structurally confined to a rank-`embed_dim` bilinear
form, which is precisely why it resists this failure mode even though it uses the *same*
embedding table. This explains both negative results above: shrinking the head (§2) and
shrinking embedding magnitude (§4 step 2) don't remove a *direction*-encoded identity
signal; only randomly dropping it out (§4 step 3, underpowered as tested) or removing the
near-uniqueness at the source (§6) would.

Note for context (corrects an earlier assumption during this project): the actual
training-set positive rate is **17.48%** (~1:4.7 imbalance, confirmed directly from
`data/processed/train.parquet`), not the ~1:1000 figure in `src/data/dataset.py`'s
module docstring (which describes the full raw Avazu dataset, not this 200k-row sample).
The near-unique-ID mechanism above, not extreme class imbalance, is the actual driver of
the Transformer's overfitting.

## 6. Results

| model | head | best epoch | val AUC | val LogLoss | test AUC | test LogLoss |
|---|---|---|---|---|---|---|
| FM (`focal_adam`) | — | 4 | 0.7642 | 0.4919 | **0.7731** | 0.4896 |
| Transformer | Flatten+MLP | 1 | 0.7314 | 0.4716 | 0.7398 | 0.4668 |
| Transformer | Mean-pool | 1 | 0.7331 | 0.4575 | 0.7387 | 0.4543 |
| Transformer | Mean-pool + embed_dropout | 1 | 0.7408 | 0.4690 | 0.7477 | 0.4664 |
| Transformer | + embedding_weight_decay (1e-1 / 1.0) | 1 | ~0.7407 | ~0.4700 | ~0.7477 | ~0.4670 |
| Transformer | + dropout=0.4 (blanket) | 1 | 0.7310 | 0.5338 | 0.7431 | 0.5316 |

Figures: `runs/comparisons/FM_vs_Transformer-flatten_vs_Transformer-meanpool_vs_Transformer-meanpool-embeddrop_{auc,gap,logloss,overview}.png`
(generated via `src/plot_results.py`, model-agnostic — no new plotting code was needed).
The `_gap.png` plot is the clearest single figure: FM's train/val AUC gap climbs
gradually across 7 self-stopped epochs, while every Transformer variant jumps to
gap ≈ 0.15-0.25 within the first 1-2 epochs and plateaus there.

**Bottom line**: FM wins on AUC (ranking quality) by ~2.5-3 points; the best Transformer
variant (mean-pool + embed_dropout) wins on LogLoss (probability calibration) by a
similar margin. Neither wins outright — which metric matters depends on whether the
downstream use case is ranking (AUC) or absolute probability estimation, e.g. bid
pricing (LogLoss).

## 7. Discussion — is this consistent with the literature?

Yes. This is a known, documented pattern, not an implementation defect:

- **AutoInt** (Song et al., 2019 — the paper this architecture is modeled on) does
  report self-attention beating FM, but trains on the *full* ~40M-row Avazu dataset
  (this project uses a 200k-row sample) and includes a residual connection carrying the
  raw embedding directly into each attention layer's output — a design choice that
  preserves low-order signal FM captures for free, which the plain stacked-encoder
  implementation here omits.
- Tabular deep-learning benchmarks (Gorishniy et al., 2021, *Revisiting Deep Learning
  Models for Tabular Data*; Shwartz-Ziv & Armon, 2022, *Tabular Data: Deep Learning is
  Not All You Need*) repeatedly find that simpler, strongly-biased models match or beat
  Transformer-style architectures on small-to-moderate tabular datasets, and that
  Transformers need more data and heavier tuning to close the gap.
  General principle: flexible, low-inductive-bias models (self-attention) need more data
  to be steered away from spurious/memorized solutions; strongly-biased closed-form
  models (FM's bilinear form) generalize better in small-sample, high-cardinality
  regimes — exactly the FM-vs-Transformer split observed here.
- Nearly every successful "Transformer for CTR" architecture in the literature (DeepFM,
  xDeepFM, AutoInt+) keeps an explicit low-order linear/FM term running in parallel with
  the attention/deep component, rather than relying on attention alone. That a *pure*
  Transformer underperforms a *pure* FM on this dataset is consistent with why that
  design choice exists in the first place.

## 8. Future work (not implemented — out of scope for this pass)

- **Feature hashing / bucketing** `device_ip` (and possibly `device_id`) down to a fixed
  number of buckets (e.g. 5,000-8,000) in `src/data/preprocess.py`, forcing many rows to
  share embedding rows and removing the near-unique-ID mechanism at the source, rather
  than regularizing around it after the fact.
- **Decoupled embed-only dropout**: give `embed_dropout` its own value in
  `TransformerCTR`, separate from the encoder/head `dropout`, so it can be pushed higher
  (§4 step 3 suggested this helps) without also suppressing attention/FFN capacity.
- **Hybrid architecture**: add FM's linear/order-1 term (or a residual embedding
  connection, as in AutoInt) alongside the Transformer's interaction output, matching
  the pattern used by every successful published Transformer-CTR model (§7).

## 9. File map

| File | Status | Purpose |
|---|---|---|
| `src/models/transformer.py` | new | `TransformerCTR` model |
| `src/models/__init__.py` | modified | exports `FactorizationMachine`, `TransformerCTR` |
| `src/train.py` | modified | `build_model` factory, `--model_type` CLI flag, param-grouped `embedding_weight_decay` in `build_optimizer` |
| `configs/config.yaml` | modified | added `model.type`, `model.transformer.*`, `train.embedding_weight_decay` (documented, opt-in) |
| `src/plot_cardinality.py` | new | field singleton-value diagnostic chart (§5) |
| `src/plot_architecture.py` | modified | added `build_transformer_figure`, `--model fm\|transformer` CLI flag |
