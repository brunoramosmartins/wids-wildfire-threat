#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# CREATE GITHUB ISSUES
# ============================================================================
# This script creates all 30 project issues for the WiDS 2026 Wildfire
# Time-to-Threat Prediction repository.
#
# Prerequisites:
#   gh auth status      (must be authenticated)
#   ./create_labels.sh  (labels must exist)
#   ./create_milestones.sh  (milestones must exist)
#
# Usage:
#   export REPO_NAME="wids-wildfire-threat"
#   chmod +x create_issues.sh
#   ./create_issues.sh
# ============================================================================

REPO_NAME="${REPO_NAME:-wids-wildfire-threat}"
GH_USER=$(gh api user --jq '.login')
REPO="${GH_USER}/${REPO_NAME}"

echo "============================================"
echo "  Creating Issues for: ${REPO}"
echo "============================================"

# Milestone titles (hardcoded to avoid WSL/Windows gh api --jq issues)
MS0="Phase 0 — Setup & Alignment"
MS1="Phase 1 — EDA & Data Understanding"
MS2="Phase 2 — Data Processing Pipeline"
MS3="Phase 3 — Feature Engineering"
MS4="Phase 4 — Baseline Model"
MS5="Phase 5 — Advanced Modeling"
MS6="Phase 6 — Optimization & Ensembles"
MS7="Phase 7 — Observability"
MS8="Phase 8 — Portfolio Polish"

# ============================================================================
# PHASE 0 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 0 — Setup & Alignment"

gh issue create --repo "${REPO}" \
  --title "feat: initialize repository and CI scaffold" \
  --milestone "${MS0}" \
  --label "type: feature,layer: infra,priority: high,phase: 0" \
  --body '## Context
The project needs a clean repository foundation before any data work begins. This includes branch protection, a CI stub, issue/PR templates, and the label/milestone taxonomy for the entire roadmap.

## Tasks
- [ ] Create repo with `README.md`, `.gitignore`, `LICENSE`
- [ ] Configure branch protection on `main`: require PR, require status check
- [ ] Create milestones for Phase 0 through Phase 8
- [ ] Create all labels (see appendix)
- [ ] Add issue templates (task + bug) and PR template
- [ ] Add minimal CI workflow (`.github/workflows/ci.yml`) with ruff + pytest placeholders

## Definition of Done
- [ ] Direct push to `main` is blocked
- [ ] All 9 milestones exist
- [ ] All labels are created
- [ ] CI runs on every PR (even if just placeholders)

## References
- [GitHub branch protection docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-a-branch-protection-rule)'

echo "  ✅ Issue #1 created"

gh issue create --repo "${REPO}" \
  --title "feat: set up Python environment, Makefile, and pre-commit" \
  --milestone "${MS0}" \
  --label "type: feature,layer: infra,priority: high,phase: 0" \
  --body '## Context
Reproducible environment setup and a single entry point for all pipeline operations. Code quality gates (ruff, mypy, pre-commit) prevent technical debt from Day 1.

## Tasks
- [ ] Create `pyproject.toml` with pinned dependencies: pandas, polars, scikit-learn, xgboost, lightgbm, lifelines, scikit-survival, mlflow, structlog, ruff, mypy, pytest, pre-commit
- [ ] Create `requirements.txt` from pyproject.toml
- [ ] Create `Makefile` with targets: `install`, `lint`, `test`, `process`, `features`, `train`, `predict`, `evaluate`, `submit`, `pipeline`
- [ ] Create `scripts/setup_env.sh`
- [ ] Configure ruff (line-length=100, Python 3.11 target) and mypy in `pyproject.toml`
- [ ] Set up `.pre-commit-config.yaml` with ruff formatter + linter hooks
- [ ] Verify `make install && make lint` passes

## Definition of Done
- [ ] `make install` works in a fresh venv
- [ ] `make lint` runs without errors on empty project
- [ ] `pre-commit run --all-files` passes
- [ ] `Makefile` has at least 10 targets

## References
- [ruff configuration](https://docs.astral.sh/ruff/configuration/)'

echo "  ✅ Issue #2 created"

gh issue create --repo "${REPO}" \
  --title "feat: download Kaggle dataset and initial inspection" \
  --milestone "${MS0}" \
  --label "type: feature,layer: data,priority: high,phase: 0" \
  --body '## Context
The Kaggle dataset (Watch Duty wildfire data) is the raw input. Automated download ensures anyone can reproduce the project. Initial inspection documents the raw data landscape.

## Tasks
- [ ] Write `scripts/download_data.sh` using `kaggle competitions download`
- [ ] Verify `train.csv`, `test.csv`, `sample_submission.csv` exist in `data/raw/`
- [ ] Log: row counts, column counts, dtypes, null percentages, unique values for categoricals
- [ ] Add data directories to `.gitignore`
- [ ] Document download instructions in `README.md`

## Definition of Done
- [ ] Script downloads and extracts all competition files
- [ ] Data files are gitignored
- [ ] Initial data stats documented

## References
- [Kaggle CLI docs](https://github.com/Kaggle/kaggle-api)
- [WiDS 2026 competition](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)'

echo "  ✅ Issue #3 created"

gh issue create --repo "${REPO}" \
  --title "docs: document problem statement and initial modeling decisions" \
  --milestone "${MS0}" \
  --label "type: docs,layer: business,priority: critical,phase: 0" \
  --body '## Context
Every subsequent phase depends on clear problem framing. The survival analysis framing, evaluation metric, and submission format must be unambiguously documented before any modeling begins.

## Tasks
- [ ] Write `docs/problem_statement.md`:
  - Competition URL, rules, deadlines
  - Target variable: definition and interpretation
  - Evaluation metric: exact formula and interpretation
  - Submission format: columns, dtypes, expected range
  - Survival analysis framing: event definition, censoring, time horizon
- [ ] Write initial `docs/modeling_decisions.md`:
  - Candidate algorithms (baseline → advanced)
  - Validation strategy (temporal vs random split)
  - Feature engineering hypotheses
  - Risks and unknowns
- [ ] Self-review: no definition uses "maybe", "usually", or "it depends"

## Definition of Done
- [ ] Both docs committed
- [ ] All rules are deterministic
- [ ] Problem framing is explicit enough that a second person could start modeling from the docs alone

## References
- [WiDS 2026 Kaggle Overview](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview)'

echo "  ✅ Issue #4 created"

# ============================================================================
# PHASE 1 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 1 — EDA & Data Understanding"

gh issue create --repo "${REPO}" \
  --title "feat: create dataset overview EDA notebook" \
  --milestone "${MS1}" \
  --label "type: feature,layer: eda,priority: high,phase: 1" \
  --body '## Context
Systematic data profiling prevents surprises downstream. Distributions, missing patterns, and target analysis inform every subsequent decision.

## Tasks
- [ ] Create `notebooks/01_eda_overview.ipynb`
- [ ] Analyze all columns: dtype, unique count, null rate, distribution (histograms for numeric, bar plots for categorical)
- [ ] Target variable deep dive: distribution, range, outliers
- [ ] Correlation matrix (numeric features)
- [ ] Missing data pattern analysis: heatmap, column-pair co-missingness
- [ ] Save key figures to `reports/figures/`

## Definition of Done
- [ ] Notebook runs end-to-end without errors
- [ ] Every column has at least one visualization
- [ ] Target variable is thoroughly characterized
- [ ] Missing patterns are documented

## References
- `data/raw/train.csv`'

echo "  ✅ Issue #5 created"

gh issue create --repo "${REPO}" \
  --title "feat: create geospatial and survival analysis EDA notebooks" \
  --milestone "${MS1}" \
  --label "type: feature,layer: eda,priority: high,phase: 1" \
  --body '## Context
This competition involves geospatial wildfire data and is framed as survival analysis. Understanding spatial distributions and time-to-event characteristics is critical for effective feature engineering.

## Tasks
- [ ] Create `notebooks/02_eda_geospatial.ipynb`:
  - Plot wildfire locations on map
  - Analyze fire-to-infrastructure distances
  - Regional aggregations
- [ ] Create `notebooks/03_eda_survival_curves.ipynb`:
  - Kaplan-Meier estimator (overall + stratified)
  - Nelson-Aalen cumulative hazard
  - Censoring rate and distribution
  - Log-rank tests for key categorical splits
- [ ] Save key figures to `reports/figures/`

## Definition of Done
- [ ] Both notebooks run end-to-end
- [ ] At least 2 geospatial visualizations saved
- [ ] Kaplan-Meier curves plotted for at least 3 feature stratifications
- [ ] Censoring analysis complete

## References
- [lifelines documentation](https://lifelines.readthedocs.io/)'

echo "  ✅ Issue #6 created"

gh issue create --repo "${REPO}" \
  --title "docs: create data dictionary" \
  --milestone "${MS1}" \
  --label "type: docs,layer: data,priority: high,phase: 1" \
  --body '## Context
A complete data dictionary ensures every team member (or future-you) understands every column without re-analyzing the raw data.

## Tasks
- [ ] Write `docs/data_dictionary.md` with table: column name, dtype, description, null rate, cardinality, notes
- [ ] Group columns by domain: identifier, geographic, temporal, weather, infrastructure, target
- [ ] Flag columns that need special handling (high cardinality, high nulls, potential leakage)

## Definition of Done
- [ ] Every column in train.csv is documented
- [ ] Grouping is logical
- [ ] Leakage risk flags are present

## References
- EDA notebooks findings'

echo "  ✅ Issue #7 created"

# ============================================================================
# PHASE 2 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 2 — Data Processing Pipeline"

gh issue create --repo "${REPO}" \
  --title "feat: define data schemas and column contracts" \
  --milestone "${MS2}" \
  --label "type: feature,layer: data,priority: high,phase: 2" \
  --body '## Context
Formal schemas prevent silent data corruption. Defining expected dtypes, required columns, and valid ranges before writing any processing code ensures the pipeline catches issues early.

## Tasks
- [ ] Create `src/data/schemas.py`
- [ ] Define `TRAIN_SCHEMA` and `TEST_SCHEMA` as dictionaries: column → dtype
- [ ] Define `REQUIRED_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`
- [ ] Define `VALID_RANGES` for bounded numeric columns

## Definition of Done
- [ ] All columns from data dictionary are covered
- [ ] Schema can be imported and used by process.py and validate.py
- [ ] Type hints on all public functions

## References
- `docs/data_dictionary.md`'

echo "  ✅ Issue #8 created"

gh issue create --repo "${REPO}" \
  --title "feat: build data processing pipeline" \
  --milestone "${MS2}" \
  --label "type: feature,layer: data,priority: critical,phase: 2" \
  --body '## Context
Raw Kaggle CSVs need cleaning before feature engineering. The processing pipeline must be deterministic, idempotent, and log all decisions for reproducibility.

## Tasks
- [ ] Create `src/data/process.py`
- [ ] Read raw CSVs from `data/raw/`
- [ ] Rename columns to `snake_case`
- [ ] Cast types per schema
- [ ] Handle nulls: document strategy per column (impute/drop/flag)
- [ ] Handle outliers: document thresholds
- [ ] Save to `data/processed/*.parquet`
- [ ] Add `make process` target to Makefile
- [ ] Log all transformations via structlog

## Definition of Done
- [ ] `make process` runs without errors
- [ ] Output Parquet files have correct dtypes
- [ ] Processing log shows all transformations applied
- [ ] Running twice produces identical output (idempotent)

## References
- `src/data/schemas.py`, `docs/data_dictionary.md`'

echo "  ✅ Issue #9 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement data validation and quality checks" \
  --milestone "${MS2}" \
  --label "type: feature,layer: data,priority: high,phase: 2" \
  --body '## Context
Automated validation catches data issues before they silently corrupt downstream models. Validation runs as a post-processing step and as standalone tests.

## Tasks
- [ ] Create `src/data/validate.py`
- [ ] Implement: schema check, null check, range check, row count check
- [ ] Hook into `process.py` (validation runs after processing)
- [ ] Create `tests/test_data_processing.py` and `tests/test_data_quality.py`
- [ ] Log validation results via structlog

## Definition of Done
- [ ] `make process` includes validation step
- [ ] Validation errors halt the pipeline with clear error messages
- [ ] At least 5 unit tests passing
- [ ] Test coverage on all validation functions

## References
- [pytest documentation](https://docs.pytest.org/)'

echo "  ✅ Issue #10 created"

# ============================================================================
# PHASE 3 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 3 — Feature Engineering"

gh issue create --repo "${REPO}" \
  --title "docs: design feature catalog" \
  --milestone "${MS3}" \
  --label "type: docs,layer: features,priority: critical,phase: 3" \
  --body '## Context
Designing features before coding them prevents aimless exploration. The catalog is the contract — only features listed here get implemented. Each feature includes a leakage risk assessment.

## Tasks
- [ ] Write `docs/feature_catalog.md` with table: feature name, source columns, computation formula, rationale, domain group, leakage assessment
- [ ] Prioritize features by expected impact (high/medium/low)
- [ ] Flag any features that require external data

## Definition of Done
- [ ] At least 20 features documented
- [ ] Every feature has a leakage assessment (safe/unsafe)
- [ ] Priority ranking is present
- [ ] Feature catalog reviewed against EDA insights

## References
- EDA notebooks, `docs/data_dictionary.md`'

echo "  ✅ Issue #11 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement geospatial and temporal feature functions" \
  --milestone "${MS3}" \
  --label "type: feature,layer: features,priority: critical,phase: 3" \
  --body '## Context
Geospatial and temporal features are the backbone of wildfire-to-infrastructure threat prediction. These two domains likely carry the most signal.

## Tasks
- [ ] Create `src/features/geospatial.py`: distance calculations, bearing, region encoding, spatial density
- [ ] Create `src/features/temporal.py`: time-of-day, duration since fire start, rolling stats, survival hazard proxies
- [ ] Type hints on all functions
- [ ] Unit tests in `tests/test_feature_engineering.py`

## Definition of Done
- [ ] Each function works on a sample DataFrame
- [ ] All functions have type hints
- [ ] At least 3 tests per module
- [ ] No future-looking features

## References
- `docs/feature_catalog.md`'

echo "  ✅ Issue #12 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement weather, infrastructure features, and build orchestrator" \
  --milestone "${MS3}" \
  --label "type: feature,layer: features,priority: high,phase: 3" \
  --body '## Context
Weather conditions and infrastructure characteristics are secondary but important feature domains. The build orchestrator combines all features into the final feature matrix.

## Tasks
- [ ] Create `src/features/weather.py`: wind, temperature, humidity aggregations
- [ ] Create `src/features/infrastructure.py`: type encoding, density, vulnerability
- [ ] Create `src/features/build.py`: orchestrator that loads processed data, applies all feature modules, saves to Parquet
- [ ] Add `make features` target to Makefile
- [ ] Verify feature matrix has expected shape and no unexpected nulls

## Definition of Done
- [ ] `make features` produces `data/features/train_features.parquet` and `test_features.parquet`
- [ ] Feature count matches catalog
- [ ] No null values in required features
- [ ] Build log shows all feature groups applied

## References
- `docs/feature_catalog.md`, `src/data/process.py`'

echo "  ✅ Issue #13 created"

gh issue create --repo "${REPO}" \
  --title "feat: create feature analysis notebook" \
  --milestone "${MS3}" \
  --label "type: feature,layer: eda,priority: medium,phase: 3" \
  --body '## Context
Post-engineering analysis validates that features have signal. Importance ranking guides which features to keep and which to drop.

## Tasks
- [ ] Create `notebooks/04_feature_analysis.ipynb`
- [ ] Correlation matrix of top features
- [ ] Mutual information scores vs target
- [ ] Feature distribution by target class (if applicable)
- [ ] Identify and flag highly correlated feature pairs (r > 0.95)

## Definition of Done
- [ ] Notebook runs end-to-end
- [ ] Top 10 features identified by importance
- [ ] Correlation clusters documented
- [ ] Updated `docs/feature_catalog.md` with keep/drop recommendations

## References
- `data/features/train_features.parquet`'

echo "  ✅ Issue #14 created"

# ============================================================================
# PHASE 4 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 4 — Baseline Model"

gh issue create --repo "${REPO}" \
  --title "feat: implement validation strategy and training orchestrator" \
  --milestone "${MS4}" \
  --label "type: feature,layer: ml,priority: critical,phase: 4" \
  --body '## Context
A sound validation strategy prevents overfitting to the public leaderboard. The training orchestrator is the central script that all models flow through.

## Tasks
- [ ] Create `src/models/train.py` with configurable validation (temporal split / K-fold)
- [ ] Read features from `data/features/train_features.parquet`
- [ ] Implement `sklearn.Pipeline` with `ColumnTransformer` for preprocessing
- [ ] Log fold-level and aggregate metrics
- [ ] Document validation strategy in `docs/modeling_decisions.md`

## Definition of Done
- [ ] `make train` runs without errors
- [ ] Validation metrics logged per fold and as mean±std
- [ ] Strategy documented with justification

## References
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)'

echo "  ✅ Issue #15 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement baseline models" \
  --milestone "${MS4}" \
  --label "type: feature,layer: ml,priority: high,phase: 4" \
  --body '## Context
Baselines set the performance floor. A simple model that works end-to-end is more valuable than a complex model that does not run.

## Tasks
- [ ] Create `src/models/baselines.py`
- [ ] Implement Logistic Regression baseline
- [ ] Implement Random Forest baseline
- [ ] Implement naive survival baseline (if applicable: Kaplan-Meier, median prediction)
- [ ] All models follow a consistent interface (fit, predict, predict_proba)

## Definition of Done
- [ ] All baselines train and predict without errors
- [ ] Each baseline has a logged MLflow run
- [ ] Baseline scores are reasonable (better than random)

## References
- `docs/problem_statement.md` (metric and target)'

echo "  ✅ Issue #16 created"

gh issue create --repo "${REPO}" \
  --title "feat: build evaluation module with MLflow tracking" \
  --milestone "${MS4}" \
  --label "type: feature,layer: ml,priority: high,phase: 4" \
  --body '## Context
Centralized evaluation ensures all models are compared on the same metric. MLflow tracking makes experiment comparison trivial.

## Tasks
- [ ] Create `src/models/evaluate.py`
- [ ] Compute competition metric + secondary metrics
- [ ] Log to MLflow: parameters, metrics, model artifact, feature list
- [ ] Generate `reports/evaluation/evaluation_{model}.md` with metrics table and interpretation
- [ ] Initialize MLflow experiment `wids-wildfire-2026`

## Definition of Done
- [ ] All metrics computed correctly
- [ ] MLflow UI (`mlflow ui`) shows experiments with logged runs
- [ ] Evaluation report is human-readable

## References
- [MLflow tracking docs](https://mlflow.org/docs/latest/tracking.html)'

echo "  ✅ Issue #17 created"

gh issue create --repo "${REPO}" \
  --title "feat: generate first Kaggle submission" \
  --milestone "${MS4}" \
  --label "type: feature,layer: submission,priority: critical,phase: 4" \
  --body '## Context
The first submission validates the entire pipeline end-to-end: data → features → model → predictions → formatted CSV → Kaggle upload. Score is secondary; pipeline correctness is the goal.

## Tasks
- [ ] Create `src/models/predict.py`: load model, predict on test features, save predictions
- [ ] Create `src/submission/format.py`: format predictions per Kaggle requirements
- [ ] Create `tests/test_submission_format.py`: validate column names, dtypes, row count, value ranges
- [ ] Generate `submissions/submission_baseline_{date}.csv`
- [ ] Submit to Kaggle and record public score
- [ ] Add `make submit` target to Makefile

## Definition of Done
- [ ] Submission file passes Kaggle validation (upload accepted)
- [ ] Public leaderboard score recorded in `docs/experiment_log.md`
- [ ] Tests validate submission format
- [ ] `make predict && make submit` works end-to-end

## References
- [WiDS 2026 submission page](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/submit)'

echo "  ✅ Issue #18 created"

# ============================================================================
# PHASE 5 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 5 — Advanced Modeling"

gh issue create --repo "${REPO}" \
  --title "feat: implement survival analysis models" \
  --milestone "${MS5}" \
  --label "type: feature,layer: ml,priority: critical,phase: 5" \
  --body '## Context
The competition is framed as survival analysis (time-to-threat prediction). Survival-specific models (Cox PH, RSF, GBS) may capture the problem structure better than generic regressors or classifiers.

## Tasks
- [ ] Create `src/models/survival.py`
- [ ] Implement Cox Proportional Hazards (lifelines)
- [ ] Implement Random Survival Forest (scikit-survival)
- [ ] Implement Gradient Boosted Survival (scikit-survival)
- [ ] Adapt `evaluate.py` for survival metrics: concordance index (C-index), Brier score, calibration
- [ ] Log all runs to MLflow

## Definition of Done
- [ ] All three models train and predict without errors
- [ ] C-index computed and logged
- [ ] Models produce risk scores compatible with submission format
- [ ] MLflow runs show hyperparameters and metrics

## References
- [lifelines docs](https://lifelines.readthedocs.io/)
- [scikit-survival docs](https://scikit-survival.readthedocs.io/)'

echo "  ✅ Issue #19 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement gradient boosting models" \
  --milestone "${MS5}" \
  --label "type: feature,layer: ml,priority: high,phase: 5" \
  --body '## Context
Gradient boosting (XGBoost, LightGBM) often dominates Kaggle leaderboards. These models handle mixed feature types well and require less feature preprocessing.

## Tasks
- [ ] Create `src/models/boosting.py`
- [ ] Implement XGBoost with appropriate objective function
- [ ] Implement LightGBM with appropriate objective function
- [ ] Implement CatBoost (optional, if categorical features are significant)
- [ ] Log all runs to MLflow with full hyperparameters

## Definition of Done
- [ ] All models train and predict without errors
- [ ] Each model logged to MLflow
- [ ] Comparison with baselines shows improvement (or documented explanation if not)

## References
- [XGBoost survival docs](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html)
- [LightGBM docs](https://lightgbm.readthedocs.io/)'

echo "  ✅ Issue #20 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement feature selection and model comparison" \
  --milestone "${MS5}" \
  --label "type: feature,layer: features,priority: medium,phase: 5" \
  --body '## Context
Too many features can cause noise and slow training. Feature selection identifies the signal. The model comparison table is the decision-making artifact for the ensemble phase.

## Tasks
- [ ] Create `src/features/selection.py`
- [ ] Implement RFE with best model
- [ ] Implement permutation importance
- [ ] Compare full feature set vs top-N features (for N = 20, 30, 50)
- [ ] Update `docs/feature_catalog.md` with keep/drop decisions
- [ ] Write model comparison table in `docs/experiment_log.md`
- [ ] Select top 2-3 models for ensemble phase
- [ ] Submit best individual model to Kaggle

## Definition of Done
- [ ] Feature selection results documented
- [ ] Model comparison table with: model, metric, training time, feature count
- [ ] Best individual model submitted to Kaggle
- [ ] Top models identified for ensemble phase

## References
- [scikit-learn RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)'

echo "  ✅ Issue #21 created"

# ============================================================================
# PHASE 6 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 6 — Optimization & Ensembles"

gh issue create --repo "${REPO}" \
  --title "feat: implement hyperparameter tuning" \
  --milestone "${MS6}" \
  --label "type: feature,layer: ml,priority: high,phase: 6" \
  --body '## Context
Default hyperparameters rarely give the best performance. Systematic tuning with proper CV ensures the search finds robust parameters, not overfit ones.

## Tasks
- [ ] Define search spaces in `configs/model_config.yaml` for top 2-3 models
- [ ] Implement tuning using Optuna or RandomizedSearchCV
- [ ] Log all trials to MLflow (including failed/early-stopped)
- [ ] Document best hyperparameters in `docs/experiment_log.md`
- [ ] Re-evaluate tuned models on validation set

## Definition of Done
- [ ] At least 50 trials per model
- [ ] Best hyperparameters documented
- [ ] Tuned model scores logged to MLflow
- [ ] Improvement over default hyperparameters is quantified

## References
- [Optuna docs](https://optuna.readthedocs.io/)'

echo "  ✅ Issue #22 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement ensemble methods" \
  --milestone "${MS6}" \
  --label "type: feature,layer: ml,priority: high,phase: 6" \
  --body '## Context
Ensembles almost always outperform individual models in Kaggle competitions. The key is combining diverse models (different algorithms, different feature subsets).

## Tasks
- [ ] Create `src/models/ensemble.py`
- [ ] Implement weighted average (optimize weights on validation set using scipy.optimize)
- [ ] Implement stacking (base models → meta-learner)
- [ ] Implement blending (holdout-based)
- [ ] Compare: best individual vs weighted avg vs stacking vs blending
- [ ] Log ensemble runs to MLflow

## Definition of Done
- [ ] All ensemble methods produce valid predictions
- [ ] Ensemble outperforms best individual model (or explanation if not)
- [ ] Weights/architecture documented
- [ ] Best ensemble submitted to Kaggle

## References
- [scikit-learn StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)'

echo "  ✅ Issue #23 created"

gh issue create --repo "${REPO}" \
  --title "feat: error analysis and final model selection" \
  --milestone "${MS6}" \
  --label "type: feature,layer: ml,priority: medium,phase: 6" \
  --body '## Context
Error analysis reveals systematic failure patterns and demonstrates analytical maturity in the portfolio. Final model selection must be justified with data.

## Tasks
- [ ] Create `notebooks/05_error_analysis.ipynb`
- [ ] Analyze: worst predictions by feature, geography, time
- [ ] Identify systematic error patterns
- [ ] Document insights in `docs/modeling_decisions.md`
- [ ] Select final model/ensemble and generate final submission
- [ ] Record final score in `docs/experiment_log.md`

## Definition of Done
- [ ] Error analysis notebook complete with visualizations
- [ ] At least 3 error patterns identified and documented
- [ ] Final model choice justified in `docs/modeling_decisions.md`
- [ ] Final submission uploaded to Kaggle

## References
- All prior models and evaluation reports'

echo "  ✅ Issue #24 created"

# ============================================================================
# PHASE 7 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 7 — Observability"

gh issue create --repo "${REPO}" \
  --title "feat: implement structured logging across pipeline" \
  --milestone "${MS7}" \
  --label "type: feature,layer: observability,priority: high,phase: 7" \
  --body '## Context
Structured logging (JSON output with context) is a production engineering practice. It makes debugging trivial and demonstrates ops awareness in a portfolio.

## Tasks
- [ ] Create `src/observability/logger.py` with structlog configuration
- [ ] Create `configs/logging_config.yaml` with log level, output format, output path
- [ ] Retrofit all `src/` modules to use structured logger
- [ ] Ensure every step logs: start time, end time, row counts, key decisions

## Definition of Done
- [ ] All scripts produce structured JSON logs
- [ ] Logs include run_id for correlation
- [ ] Log levels are appropriate (not everything is INFO)
- [ ] `make pipeline` produces a full log trace

## References
- [structlog docs](https://www.structlog.org/)'

echo "  ✅ Issue #25 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement data quality monitoring and feature drift detection" \
  --milestone "${MS7}" \
  --label "type: feature,layer: observability,priority: medium,phase: 7" \
  --body '## Context
Data quality monitoring catches silent data corruption. Feature drift between train and test distributions can explain poor generalization.

## Tasks
- [ ] Create `src/observability/data_quality.py`
- [ ] Implement data profiling: null rates, distribution stats, outlier counts
- [ ] Generate baseline profile and comparison profile
- [ ] Create `src/observability/feature_drift.py`
- [ ] Implement PSI for numeric features, chi-square for categorical
- [ ] Generate drift report with alert thresholds
- [ ] Integrate into `make process` and `make features` steps

## Definition of Done
- [ ] Data quality profile generated after processing
- [ ] Drift report generated after feature engineering
- [ ] Alerts fire when PSI > 0.2
- [ ] Reports saved to `reports/data_quality/`

## References
- [PSI reference](https://www.listendata.com/2015/05/population-stability-index.html)'

echo "  ✅ Issue #26 created"

gh issue create --repo "${REPO}" \
  --title "feat: implement pipeline health checks" \
  --milestone "${MS7}" \
  --label "type: feature,layer: observability,priority: medium,phase: 7" \
  --body '## Context
Pipeline health checks are the automated safety net. They verify that each step produced expected outputs before proceeding.

## Tasks
- [ ] Create `src/observability/pipeline_health.py`
- [ ] Implement file existence checks after each step
- [ ] Implement row count consistency checks
- [ ] Implement prediction range validation
- [ ] Implement submission file validation
- [ ] Integrate into `make pipeline` (health check between each step)

## Definition of Done
- [ ] Pipeline halts with clear error if a health check fails
- [ ] All steps have at least one health check
- [ ] `make pipeline` includes health checks between steps

## References
- Existing `src/data/validate.py`'

echo "  ✅ Issue #27 created"

# ============================================================================
# PHASE 8 ISSUES
# ============================================================================
echo ""
echo ">>> Phase 8 — Portfolio Polish"

gh issue create --repo "${REPO}" \
  --title "docs: write portfolio-grade README and runbook" \
  --milestone "${MS8}" \
  --label "type: docs,layer: infra,priority: critical,phase: 8" \
  --body '## Context
The README is the first thing a recruiter or senior engineer sees. It must communicate the project value in 30 seconds and provide a clear path to reproduce results.

## Tasks
- [ ] Write `README.md` with: overview, architecture diagram, quick start, results, tech stack
- [ ] Write `docs/runbook.md` with: execution guide, benchmarks, troubleshooting
- [ ] Add tech stack badges (Python, CI status, license)
- [ ] Link to Kaggle competition and leaderboard position

## Definition of Done
- [ ] `README.md` answers: what, why, how, results — in under 2 minutes of reading
- [ ] `docs/runbook.md` lets someone run the pipeline with zero prior knowledge
- [ ] Benchmarks include actual timing measurements

## References
- [Awesome README examples](https://github.com/matiassingers/awesome-readme)'

echo "  ✅ Issue #28 created"

gh issue create --repo "${REPO}" \
  --title "feat: finalize pipeline orchestration and CI" \
  --milestone "${MS8}" \
  --label "type: feature,layer: infra,priority: high,phase: 8" \
  --body '## Context
A portfolio project must run end-to-end with a single command. CI must validate that the pipeline does not silently break.

## Tasks
- [ ] Create/update `scripts/run_pipeline.sh` with full cycle: process → features → train → predict → evaluate → submit
- [ ] Update `Makefile` with `pipeline` target
- [ ] Add timing instrumentation
- [ ] Update CI: lint + test + smoke run on sample data
- [ ] Verify `make pipeline` from clean state (after `make install` + data download)

## Definition of Done
- [ ] `make pipeline` runs end-to-end without errors
- [ ] CI passes on main branch
- [ ] Each step timing is logged

## References
- All prior phases'

echo "  ✅ Issue #29 created"

gh issue create --repo "${REPO}" \
  --title "docs: create experiment log and CHANGELOG" \
  --milestone "${MS8}" \
  --label "type: docs,layer: business,priority: high,phase: 8" \
  --body '## Context
The experiment log tells the story of the ML journey — what was tried, what worked, what did not. The CHANGELOG summarizes the project evolution.

## Tasks
- [ ] Write `docs/experiment_log.md`:
  - Chronological log of key experiments
  - Results table: model, features, metric, date
  - Insights: what worked and what did not
  - Final model architecture and justification
- [ ] Create `CHANGELOG.md` covering all 9 phases
- [ ] Final review of all docs for consistency

## Definition of Done
- [ ] Experiment log has at least 10 entries
- [ ] CHANGELOG covers every phase
- [ ] All docs cross-reference correctly

## References
- MLflow runs, evaluation reports'

echo "  ✅ Issue #30 created"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================"
echo "  ✅ ALL 30 ISSUES CREATED"
echo "============================================"
echo ""
echo "  Repository: https://github.com/${REPO}"
echo "  Issues:     30 created"
echo ""
echo "  Phase 0: Issues #1-#4   (Setup)"
echo "  Phase 1: Issues #5-#7   (EDA)"
echo "  Phase 2: Issues #8-#10  (Processing)"
echo "  Phase 3: Issues #11-#14 (Features)"
echo "  Phase 4: Issues #15-#18 (Baseline)"
echo "  Phase 5: Issues #19-#21 (Advanced)"
echo "  Phase 6: Issues #22-#24 (Optimization)"
echo "  Phase 7: Issues #25-#27 (Observability)"
echo "  Phase 8: Issues #28-#30 (Polish)"
echo ""
echo "  Next step: git checkout -b phase/0-setup"
echo "============================================"
