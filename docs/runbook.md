# Runbook

## Prerequisites

- Python 3.11+
- Kaggle CLI configured (`kaggle` command available)
- Git

## Quick Start

```bash
# 1. Set up environment
bash scripts/setup_env.sh

# 2. Download data
bash scripts/download_data.sh

# 3. Run full pipeline
make pipeline
```

## Step-by-Step Execution

| Step | Command | Input | Output |
|------|---------|-------|--------|
| Install | `make install` | `requirements.txt` | Virtual environment ready |
| Process | `make process` | `data/raw/*.csv` | `data/processed/*.parquet` |
| Features | `make features` | `data/processed/*.parquet` | `data/features/*.parquet` |
| Train | `make train` | `data/features/train_features.parquet` | Model in `models/` |
| Predict | `make predict` | Model + test features | `data/predictions/*.parquet` |
| Evaluate | `make evaluate` | Predictions | `reports/evaluation/*.md` |
| Submit | `make submit` | Predictions | `submissions/*.csv` |

## Benchmarks

<!-- Timing measurements for each step -->

| Step | Duration | Notes |
|------|----------|-------|
| Process | | |
| Features | | |
| Train | | |
| Predict | | |
| Evaluate | | |
| Total | | |

## Troubleshooting

<!-- Common issues and solutions -->
