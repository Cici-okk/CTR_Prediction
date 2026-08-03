# Getting Started (FM baseline + training pipeline)

Quick orientation for teammates picking this up: what got added since the data pipeline
(Member A) was finished, and how to run it.

## What's new

- **`src/models/fm.py`** — a Factorization Machine (FM) baseline model.
- **`src/train.py`** — a training script: BCE vs Focal loss, Adam vs AdamW, Dropout,
  Early Stopping, LR Warmup, AUC/LogLoss tracked every epoch, all config- and
  CLI-driven (no code edits needed to run a different combination).
- **`src/plot_results.py`** — compares two or more training runs: a metrics table plus
  train/val curve and train/val-gap plots.
- **`configs/config.yaml`** — added a `model:` section (`embed_dim`, `dropout`) and
  `train.early_stopping_metric`.
- **`src/utils.py`** — added `compute_metrics` (AUC/LogLoss) and an `EarlyStopping` helper.

Nothing under `src/data/` or `data/processed/` changed — the dataloaders and the locked
`meta.json`/parquet files are exactly as Member A left them.

## Setup

```bash
source .venv/bin/activate        # the venv already has everything in requirements.txt
```

If you're on a fresh machine without the `.venv/` in this repo:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No data setup needed — `data/processed/` is already committed and ready to use.

## Run it

**Smoke test** (~5 seconds, just confirms everything wires up):
```bash
python -m src.train --output_dir runs/smoke --epochs 1
```

**A single full training run** (defaults come from `configs/config.yaml`: focal loss,
AdamW, dropout 0.2, early stopping on val AUC with patience 3):
```bash
python -m src.train --output_dir runs/my_run
```

**CLI overrides** (for ablations — no yaml editing needed):
```bash
python -m src.train --loss bce               --output_dir runs/bce_adamw
python -m src.train --optimizer adam         --output_dir runs/focal_adam
python -m src.train --dropout 0.0 --no_early_stop --epochs 20 --output_dir runs/no_reg
```

Each run writes to its `--output_dir`: `best_model.pt`, `history.json` (per-epoch
metrics), `metrics.json` (final summary), `config_used.yaml` (exact config for that
run, for reproducibility). Everything under `runs/` is gitignored — don't commit it.

**Compare runs / generate plots:**
```bash
python -m src.plot_results --runs runs/bce_adamw runs/focal_adamw --labels bce focal --out_dir runs/comparisons
```
Prints + saves a comparison table (`{tag}_table.csv`) and three PNGs: `{tag}_auc.png`,
`{tag}_logloss.png` (train vs val curves), `{tag}_gap.png` (train−val AUC gap per
epoch — the overfitting diagnosis).

## Where to plug in

- **Building a different model** (e.g. Transformer/DeepFM): follow the same interface as
  `FactorizationMachine.forward(categorical) -> (B,) raw logits` — `src/train.py` doesn't
  care what's inside as long as that contract holds. Swap the `FactorizationMachine(...)`
  construction in `src/train.py` for your model.
- **Loading data yourself**: `from src.data.dataset import build_dataloader` — see the
  main `README.md`'s "Loading the data" section, unchanged.
- **Full technical writeup** (model math, design decisions, actual results): see
  `IMPLEMENTATION_NOTES.md`.
