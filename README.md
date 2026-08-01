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
│   ├── models/              # Model architectures (Transformer / DeepFM etc., owned by Member B)
│   ├── train.py             # Training script (tuning experiments owned by Member C)
│   └── utils.py             # Common utilities (metrics, seed, etc.)
├── configs/
│   └── config.yaml          # Hyperparameter configuration
├── notebooks/                # Experiment / visualization notebooks
├── reports/                  # Final report and slides (owned by Member D)
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
(`train.parquet` / `val.parquet` / `test.parquet` + `meta.json`). Members B/C/D can use it
directly (see "Loading the data" below) without downloading anything.

Actual stats of this subset:
- label column: `click`, positive rate ≈ 17.4% (neg:pos ≈ 4.7:1 — moderately imbalanced, not the
  1:1000 originally assumed in the pitch)
- 22 categorical features (`site_id`, `device_id`, `device_ip`, `C1`/`C14`-`C21`, etc.), no numeric features
- `device_ip` alone has ~76k distinct values in just this 200k-row sample — a high-cardinality
  feature Member B should account for in the embedding layer (e.g. hashing trick or bucketing rare IDs)

**Do not re-run `preprocess.py` with a different `--nrows` and commit over these files** — the
label encoding (`vocab_sizes` in `meta.json`) is derived from whichever rows are read, so a
different sample size changes every categorical column's vocab size and index mapping, breaking
compatibility with whatever embedding dimensions Member B has already built against `meta.json`.
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

## Member A Notes

1. `src/data/preprocess.py`: reads raw data -> handles missing values -> encodes categorical features -> splits into train/val/test, writing output to `data/processed/`.
2. `src/data/dataset.py`: `CTRDataset` (PyTorch Dataset) + an optional resampling `DataLoader` wrapper for handling extreme class imbalance.
3. `src/losses/focal_loss.py`: `FocalLoss` implementation, plus an interface to compare it against standard `BCEWithLogitsLoss`.

To run the pipeline:

```bash
python -m src.data.preprocess --raw_path data/raw/train.csv --out_dir data/processed
python -m src.losses.focal_loss  # built-in __main__ for a numerical sanity check / comparison
```
