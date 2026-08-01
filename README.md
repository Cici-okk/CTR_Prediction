# Ad Recommendation System - CTR Prediction

Predicting ad click-through rate (CTR) from user behavior history, addressing extreme class imbalance (~1:1000) and long user behavior sequences.

## Project Structure

```
project/
├── data/
│   ├── raw/            # Raw data (not tracked in git, download and place here)
│   └── processed/      # Preprocessed data
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

A public CTR dataset is recommended (e.g. Criteo / Avazu). Fields typically include:
- `label`: whether the ad was clicked (0/1, positive:negative ratio is usually ~1:1000)
- Numeric features: several continuous features
- Categorical features: several high-cardinality categorical features (require label encoding / embeddings)

Place the raw data under `data/raw/` (this directory is already ignored by `.gitignore` and will not be committed).

## Member A Notes

1. `src/data/preprocess.py`: reads raw data -> handles missing values -> encodes categorical features -> splits into train/val/test, writing output to `data/processed/`.
2. `src/data/dataset.py`: `CTRDataset` (PyTorch Dataset) + an optional resampling `DataLoader` wrapper for handling extreme class imbalance.
3. `src/losses/focal_loss.py`: `FocalLoss` implementation, plus an interface to compare it against standard `BCEWithLogitsLoss`.

To run the pipeline:

```bash
python -m src.data.preprocess --raw_path data/raw/train.csv --out_dir data/processed
python -m src.losses.focal_loss  # built-in __main__ for a numerical sanity check / comparison
```
