# WiDS 2026 — Wildfire Time-to-Threat Prediction

> Predicting Time-to-Threat for Evacuation Zones Using Survival Analysis

[![CI](https://github.com/brunoramosmartins/wids-wildfire-threat/actions/workflows/ci.yml/badge.svg)](https://github.com/brunoramosmartins/wids-wildfire-threat/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

End-to-end ML pipeline for the [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26) — predicting when active wildfires will threaten critical infrastructure using survival analysis on real-time wildfire data from Watch Duty.

## Quick Start

```bash
# 1. Set up environment
bash scripts/setup_env.sh

# 2. Download Kaggle data
bash scripts/download_data.sh

# 3. Run full pipeline
make pipeline
```

## Pipeline

```
CSV → process → features → train → predict → evaluate → submit
```

| Step | Command | Input | Output |
|------|---------|-------|--------|
| Process | `make process` | `data/raw/*.csv` | `data/processed/*.parquet` |
| Features | `make features` | `data/processed/*.parquet` | `data/features/*.parquet` |
| Train | `make train` | Feature matrices | Model artifacts |
| Predict | `make predict` | Model + test features | Predictions |
| Evaluate | `make evaluate` | Predictions | Evaluation reports |
| Submit | `make submit` | Predictions | `submissions/*.csv` |

## Project Structure

```
wids-wildfire-threat/
├── .github/              # CI, issue & PR templates
├── configs/              # Model, data, and logging config (YAML)
├── data/                 # Raw, processed, features, predictions (gitignored)
├── docs/                 # Problem statement, data dictionary, feature catalog
├── models/               # Serialized model artifacts (gitignored)
├── mlruns/               # MLflow tracking (gitignored)
├── notebooks/            # EDA and analysis notebooks
├── reports/              # Figures, evaluation reports, data quality
├── scripts/              # Setup, download, pipeline, GitHub automation
├── src/                  # Source code
│   ├── data/             # Processing, validation, schemas
│   ├── features/         # Feature engineering by domain
│   ├── models/           # Training, prediction, evaluation, model implementations
│   ├── submission/       # Kaggle submission formatting
│   ├── observability/    # Logging, data quality, drift detection
│   └── utils/            # I/O, config, reproducibility
├── submissions/          # Kaggle submission CSVs (gitignored)
├── tests/                # Unit and integration tests
├── Makefile              # Pipeline orchestration
└── pyproject.toml        # Single source of truth: deps, extras, ruff, mypy, pytest
```

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| ML (classical) | scikit-learn, XGBoost, LightGBM |
| ML (survival) | lifelines, scikit-survival |
| Data | pandas, polars, Parquet |
| Experiment tracking | MLflow (local) |
| Logging | structlog |
| Code quality | ruff, mypy, pre-commit |
| Testing | pytest |
| CI | GitHub Actions |

## Development

```bash
make install    # pip install -e ".[dev,notebook]" + pre-commit hooks
make lint       # ruff + mypy
make test       # pytest
make clean      # Remove __pycache__
```

### Dependency extras

All Python dependencies are declared in `pyproject.toml` (no `requirements.txt`). Three extras groups:

| Extra | Who needs it | Includes |
|-------|--------------|----------|
| (core) | Everyone | pandas, scikit-learn, XGBoost, lightgbm, catboost, lifelines, scikit-survival, MLflow, structlog, optuna, scipy, matplotlib, seaborn, plotly, pyarrow, pyyaml |
| `[dev]` | Contributors + CI | ruff, mypy, pytest, pre-commit, types-PyYAML |
| `[notebook]` | Local EDA/analysis | ipykernel, jupyter, nbconvert |
| `[all]` | `[dev] + [notebook]` | everything |

Examples:

```bash
pip install -e ".[dev]"            # lean CI install (no Jupyter)
pip install -e ".[dev,notebook]"   # full local dev (default, used by make install)
pip install -e ".[all]"            # same as above
```

## License

MIT
