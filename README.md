# Ad Recommendation System - CTR Prediction

Predicting ad click-through rate (CTR) from user behavior history, addressing class imbalance and long user behavior sequences. (Note: the original pitch assumed ~1:1000 imbalance; the actual dataset we're using — Avazu — is closer to 1:4.7, see the Dataset section below.)

## Project Structure

```
project/
├── data/
│   ├── raw/            # Raw data (not tracked in git; only needed if you want to regenerate data/processed)
│   └── processed/      # Preprocessed Avazu subset (200k rows), tracked in git - ready to use as-is
├── src/
│   ├── data/
│   │   ├── preprocess.py   # Data cleaning, feature encoding, train/val/test split
│   │   └── dataset.py      # PyTorch Dataset / DataLoader with sampling strategies
│   ├── losses/
│   │   └── focal_loss.py   # Focal Loss implementation, compared with BCE
│   ├── models/
│   │   ├── fm.py           # Factorization Machine baseline
│   │   └── transformer.py  # Transformer (self-attention) CTR model
│   ├── train.py             # Training script (model-agnostic: FM or Transformer via --model_type)
│   ├── plot_results.py      # Run comparison: metrics table + train/val curve + gap plots
│   ├── plot_architecture.py # Static architecture diagrams (FM / Transformer)
│   ├── plot_cardinality.py  # Field cardinality / near-unique-ID diagnostic chart
│   └── utils.py             # Common utilities (metrics, seed, early stopping, etc.)
├── configs/
│   └── config.yaml          # Hyperparameter configuration
├── doc/
│   ├── FM/                  # FM baseline: implementation notes, slide outline, script
│   └── Transformer/         # Transformer model: implementation notes + overfitting diagnosis
├── notebooks/                # Experiment / visualization notebooks
├── reports/                  # Final report, slides, and generated figures
├── requirements.txt
└── README.md
```

## Team Assignments

| Member | Core Task | Deliverable |
|---|---|---|
| A | Data & Loss | Handle imbalanced data, implement Focal Loss and the data pipeline |
| B | Model Building | Transformer/Self-Attention module and Target Attention optimization |
| C | Tuning & Experiments | Adam vs AdamW comparison, AUC evaluation, Loss/ROC curve plots |
| D | LLM Use Case & Docs | LLM application for Lead Ads, compile a 5-10 page report and slides |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

We use the [Avazu CTR Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction) Kaggle competition dataset.

**`data/processed/` is already committed to this repo** — a preprocessed 200k-row subset
(`train.parquet` / `val.parquet` / `test.parquet` + `meta.json`). Everyone can use it
directly (see "Loading the data" below) without downloading anything.

Actual stats of this subset:
- label column: `click`, positive rate ≈ 17.4% (neg:pos ≈ 4.7:1 — moderately imbalanced, not the
  1:1000 originally assumed in the pitch)
- 22 categorical features (`site_id`, `device_id`, `device_ip`, `C1`/`C14`-`C21`, etc.), no numeric features
- `device_ip` alone has ~76.7k distinct values in just this 200k-row sample, of which **26.9% of
  training rows** have a `device_ip` value that appears nowhere else in the training set — a
  near-unique-ID problem that turned out to matter a lot for the Transformer model (see
  Results & Analysis below)

**Do not re-run `preprocess.py` with a different `--nrows` and commit over these files** — the
label encoding (`vocab_sizes` in `meta.json`) is derived from whichever rows are read, so a
different sample size changes every categorical column's vocab size and index mapping, breaking
compatibility with whatever embedding dimensions have already been built against `meta.json`.
If the team decides to scale up later, coordinate first.

If you do need the raw data (e.g. to regenerate `data/processed/` at a different size), download
it yourself via `kagglehub` (requires your own Kaggle account + accepting the competition rules) —
raw data isn't tracked in git both because of its size (~1GB compressed) and because redistributing
Kaggle competition data is against its terms:

```python
import kagglehub
path = kagglehub.competition_download('avazu-ctr-prediction')
```

Then symlink `train.gz` from that path into `data/raw/train.gz`.

## Loading the data

```python
from src.data.dataset import build_dataloader

train_loader = build_dataloader("data/processed", "train", batch_size=1024, strategy="class_weight")
batch = next(iter(train_loader))
# batch["numeric"]:     (B, 0)  -- Avazu has no numeric features
# batch["categorical"]: (B, 22) -- label-encoded categorical features, see meta.json for column order / vocab_sizes
# batch["label"]:       (B,)    -- 0/1 click label
```

## Models

Both models share the same interface — `forward(categorical) -> (B,) raw logits`
(pre-sigmoid, so both plug directly into `nn.BCEWithLogitsLoss` and `FocalLoss`
unchanged) — and the same required constructor signature
(`vocab_sizes, categorical_cols, embed_dim=16, dropout=0.2`), so `src/train.py` can
build either one from a single `--model_type fm|transformer` flag without any other
code changes.

### Factorization Machine (`src/models/fm.py`)

Standard 2-way FM over the 22 categorical fields:

```
logit = bias + sum_i w(x_i) + 0.5 * sum_k[(sum_i v_i,k(x_i))^2 - sum_i v_i,k(x_i)^2]
```

All fields share one embedding table, indexed via per-field offsets (99,149 total
vocab). Full design rationale (initialization, why AdamW over Adam, dropout placement,
early-stopping metric choice) is in `doc/FM/IMPLEMENTATION_NOTES.md`.

### Transformer (`src/models/transformer.py`)

AutoInt-style feature interaction: each field's embedding is one token in a length-22
sequence, self-attention over that sequence learns interactions automatically instead
of FM's closed-form pairwise term:

```
tokens = dropout(field_embedding(x) + field_positional_embedding)   # (B, 22, d)
tokens = TransformerEncoder(tokens)                                   # (B, 22, d)
logit  = Linear(dropout(mean_pool(tokens)))                          # (B,)
```

Same shared-embedding-table mechanism as FM, plus a learnable per-field positional
embedding (needed because self-attention is permutation-invariant but the 22 fields
are not interchangeable). The head is mean-pool + a single `Linear`, not flatten+MLP —
that choice was the result of an empirical ablation, see Results & Analysis below.

## Running training

**Smoke test** (~5 seconds, just confirms everything wires up):
```bash
python -m src.train --output_dir runs/smoke --epochs 1
```

**A single full training run** (defaults come from `configs/config.yaml`: FM, focal
loss, AdamW, dropout 0.2, early stopping on val AUC with patience 3):
```bash
python -m src.train --output_dir runs/my_run
```

**Train the Transformer instead:**
```bash
python -m src.train --model_type transformer --output_dir runs/my_transformer_run
```

**CLI overrides** (for ablations — no yaml editing needed):
```bash
python -m src.train --loss bce               --output_dir runs/bce_adamw
python -m src.train --optimizer adam         --output_dir runs/focal_adam
python -m src.train --dropout 0.0 --no_early_stop --epochs 20 --output_dir runs/no_reg
python -m src.train --model_type transformer --embedding_weight_decay 1e-1 --output_dir runs/transformer_embwd
```

Each run writes to its `--output_dir`: `best_model.pt`, `history.json` (per-epoch
metrics), `metrics.json` (final summary), `config_used.yaml` (exact config for that
run, for reproducibility). Everything under `runs/` is gitignored — don't commit it.

**Compare runs / generate plots** (model-agnostic — works for any mix of FM/Transformer runs):
```bash
python -m src.plot_results --runs runs/focal_adam runs/my_transformer_run \
  --labels FM Transformer --out_dir runs/comparisons --overview
```
Prints + saves a comparison table (`{tag}_table.csv`) and PNGs: `{tag}_auc.png`,
`{tag}_logloss.png` (train vs val curves), `{tag}_gap.png` (train−val AUC gap per
epoch — the overfitting diagnosis), and with `--overview`, `{tag}_overview.png` (final
test AUC/LogLoss bars).

**Architecture diagrams:**
```bash
python -m src.plot_architecture --model fm            # reports/figures/fm_architecture.png
python -m src.plot_architecture --model transformer   # reports/figures/transformer_architecture.png
```

**Field cardinality diagnostic** (singleton-value fraction per high-cardinality field,
the evidence behind the Transformer's overfitting root cause):
```bash
python -m src.plot_cardinality --out reports/figures/field_cardinality.png
```

## Results & Analysis: FM vs Transformer

| model | head | best epoch | val AUC | val LogLoss | test AUC | test LogLoss |
|---|---|---|---|---|---|---|
| FM (`focal_adam`) | — | 4 | 0.7642 | 0.4919 | **0.7731** | 0.4896 |
| Transformer | Flatten+MLP | 1 | 0.7314 | 0.4716 | 0.7398 | 0.4668 |
| Transformer | Mean-pool | 1 | 0.7331 | 0.4575 | 0.7387 | 0.4543 |
| Transformer | Mean-pool + embed_dropout | 1 | 0.7408 | 0.4690 | 0.7477 | 0.4664 |
| Transformer | + embedding_weight_decay (1e-1 / 1.0) | 1 | ~0.7407 | ~0.4700 | ~0.7477 | ~0.4670 |
| Transformer | + dropout=0.4 (blanket) | 1 | 0.7310 | 0.5338 | 0.7431 | 0.5316 |

**FM wins on AUC** (ranking quality) by ~2.5-3 points; **the best Transformer variant
wins on LogLoss** (probability calibration) by a similar margin. Which one "wins"
depends on the downstream use case (ranking vs. absolute probability estimation, e.g.
bid pricing).

### Why the Transformer overfits harder — the diagnosis, step by step

1. **Head capacity (Flatten+MLP → mean-pool): ruled out.** The first head design
   flattened all 22 field representations into one big vector before an MLP, which
   gives the head a separate weight block per field *position* — enough capacity to
   memorize specific field-value combinations. Switching to mean-pool + a single
   `Linear` removed that capacity, but the train/val AUC gap at epoch 4 barely changed
   (0.236 → 0.246) and test AUC was statistically unchanged. Head capacity was not the
   driver.
2. **Missing embedding dropout: a real (if modest) fix.** FM applies dropout directly to
   the field embeddings before its interaction sum; the Transformer's tokens went
   straight into the encoder with no equivalent. Adding `embed_dropout` improved test
   AUC from 0.7387 to 0.7477.
3. **`embedding_weight_decay` sweep (1e-4 / 1e-1 / 1.0): ruled out, with an explanation.**
   Giving the embedding table its own, much larger weight decay produced virtually
   identical results at every magnitude tested. Reason: `nn.TransformerEncoderLayer`
   applies LayerNorm right after each residual add, and the trainable Q/K/V projections
   between the embedding table and that LayerNorm can freely rescale to compensate for
   any magnitude shrinkage weight decay imposes — weight decay constrains vector
   *magnitude*, but the exploited signal (see below) lives in vector *direction*, which
   magnitude decay can't touch.
4. **Blanket `dropout=0.4`: partial signal, wrong tool.** This measurably slowed early
   memorization (epoch-1 train/val gap dropped from ~0.16-0.19 to 0.047), confirming
   that randomly zeroing activations — unlike shrinking weight magnitude — can interfere
   with the mechanism. But applying it uniformly to embed/attention/FFN/head at once cost
   too much capacity: the gap climbed back to 0.252 by epoch 4 anyway, and test LogLoss
   got much worse (0.5316 vs 0.4664). Right direction, wrong granularity.

### Root cause: near-unique categorical IDs

`reports/figures/field_cardinality.png` shows, for each high-cardinality field, what
fraction of training rows have a value that appears **exactly once** in the training set:

| field | vocab size | % of training rows with a singleton value |
|---|---|---|
| `device_ip` | 76,668 | **26.9%** |
| `device_id` | 15,040 | 6.1% |
| `device_model` | 3,104 | 0.4% |
| `site_domain` / `site_id` / `app_id` | ~1,000-1,200 | 0.2% |

For over a quarter of training rows, `device_ip` is effectively a row ID, not a
generalizable category. Self-attention's query/key mechanism is structurally suited to
exploiting this — it can learn to use a field's embedding as a key that near-uniquely
identifies a training row and route that row's label straight to the output. FM's
closed-form pairwise term has no mechanism to express "if this exact vector then this
label"; it's structurally confined to a rank-`embed_dim` bilinear form, which is exactly
why it resists this failure mode despite using the *same* embedding table. This is also
why shrinking the head (step 1) and shrinking embedding magnitude (step 3) didn't help —
neither removes a *direction*-encoded identity signal.

### Is this consistent with the literature? Yes.

- **AutoInt** (Song et al., 2019 — the paper this architecture is modeled on) reports
  self-attention beating FM, but trains on the full ~40M-row Avazu dataset (this project
  uses a 200k-row sample) and adds a residual connection carrying the raw embedding
  directly into each attention layer's output, preserving low-order signal that the
  plain stacked-encoder version here omits.
- Tabular deep-learning benchmarks (Gorishniy et al. 2021; Shwartz-Ziv & Armon 2022)
  repeatedly find that simpler, strongly-biased models match or beat Transformer-style
  architectures on small-to-moderate tabular datasets — flexible, low-inductive-bias
  models need more data to be steered away from memorized solutions.
- Nearly every published "Transformer for CTR" architecture (DeepFM, xDeepFM, AutoInt+)
  keeps an explicit low-order linear/FM term running in parallel with the attention
  component, rather than relying on attention alone — exactly because pure attention
  alone is known to underperform on this kind of data.

**Future work** (not implemented — out of scope for this pass): feature hashing/bucketing
`device_ip`/`device_id` in `src/data/preprocess.py` to remove the near-unique-ID
mechanism at the source; a decoupled embed-only dropout rate (separate from
encoder/head dropout); or a hybrid architecture that adds FM's linear term alongside the
Transformer's interaction output, matching the pattern used by every successful
published Transformer-CTR model.

Full technical writeups (more detail, all intermediate results, code snippets) live in
`doc/FM/IMPLEMENTATION_NOTES.md` and `doc/Transformer/IMPLEMENTATION_NOTES.md`.

## Extending this further

Adding another model (e.g. DeepFM) means implementing the same contract —
`forward(categorical) -> (B,) raw logits`, constructor signature
`(vocab_sizes, categorical_cols, embed_dim=16, dropout=0.2)` — and adding one branch to
`build_model()` in `src/train.py`. Nothing else in the training loop, checkpointing, or
evaluation needs to change.

## Data pipeline notes (Member A)

1. `src/data/preprocess.py`: reads raw data -> handles missing values -> encodes categorical features -> splits into train/val/test, writing output to `data/processed/`.
2. `src/data/dataset.py`: `CTRDataset` (PyTorch Dataset) + an optional resampling `DataLoader` wrapper for handling extreme class imbalance.
3. `src/losses/focal_loss.py`: `FocalLoss` implementation, plus an interface to compare it against standard `BCEWithLogitsLoss`.

To run the pipeline:

```bash
python -m src.data.preprocess --raw_path data/raw/train.csv --out_dir data/processed
python -m src.losses.focal_loss  # built-in __main__ for a numerical sanity check / comparison
```
