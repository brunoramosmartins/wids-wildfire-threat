# Modeling Decisions

## Problem Framing

- **Task:** Predict the probability that an active wildfire will hit critical infrastructure within T hours (T = 12, 24, 48, 72)
- **Framing:** Survival analysis — time-to-event with right censoring
- **Target:** `time_to_hit_hours` (continuous, [0, 72]) + `event` (binary, 1=hit, 0=censored)
- **Output:** P(hit by T) = 1 - S(T) for each test event at T = 12, 24, 48, 72
- **Evaluation metric:** TBD from competition page

## Dataset Characteristics (EDA-informed)

- **Very small dataset:** 221 training samples, 95 test samples
- **High censoring rate:** 68.8% of fires never reach the evacuation zone
- **Zero null values:** No missing data handling needed
- **High feature sparsity:** Many features are >50% zeros (fires with no observed movement)
- **Highly correlated features:** Multiple redundant pairs (e.g., `area_growth_rel_0_5h` ~ `relative_growth_0_5h`)
- **All numeric features:** No categorical encoding needed

## Key Predictors (from EDA)

1. **Distance features** — `dist_min_ci_0_5h` is the strongest discriminator between hit/censored events
2. **Closing speed** — `closing_speed_m_per_h` shows clear separation in survival curves
3. **Directionality** — `alignment_abs` indicates whether fire is moving toward evacuation zone
4. **Fire size** — `log1p_area_first` has moderate predictive power
5. **Temporal metadata** — `event_start_hour`, `event_start_month` show some patterns but small sample sizes

## Candidate Algorithms

### Baselines
- Logistic Regression (for binary at each horizon)
- Random Forest (for binary at each horizon)

### Survival Models (preferred)
- **Cox Proportional Hazards** (lifelines) — interpretable, handles censoring natively
- **Random Survival Forest** (scikit-survival) — non-parametric, handles non-linear effects
- **Gradient Boosted Survival** (scikit-survival) — best of boosting + survival

### Gradient Boosting
- XGBoost with `survival:aft` objective
- LightGBM (regression on time, or classification at each horizon)

### Ensembles
- Weighted average of survival model outputs
- Stacking with logistic meta-learner

## Validation Strategy

- **Approach:** Stratified K-fold (K=5) stratified on `event` to maintain censoring ratio per fold
- **Rationale:** With only 221 samples, temporal split would leave too few samples per fold. Stratification ensures each fold has similar event/censored ratio (~31% hit rate per fold).
- **Primary metric:** Mean Brier Score across 4 horizons (12h, 24h, 48h, 72h)
- **Secondary metrics:** Per-horizon AUC, per-horizon log loss
- **Risk:** Small dataset means high variance in CV estimates — report mean +/- std

## Baseline Models (Phase 4)

Three baselines establish the performance floor:

1. **Kaplan-Meier baseline** — predicts the same marginal P(hit by T) for all samples. No features used. This is the "naive" baseline that any model must beat.
2. **Logistic Regression (per-horizon)** — trains 4 independent classifiers, one per horizon. StandardScaler preprocessing. Simple and interpretable.
3. **Random Forest (per-horizon)** — same per-horizon approach with RandomForestClassifier. Captures non-linear effects without tuning.

All baselines use a `MultiHorizonClassifier` wrapper that enforces monotonicity: P(hit by 12h) <= P(hit by 24h) <= P(hit by 48h) <= P(hit by 72h).

## MLflow Tracking

- **Experiment:** `wids-wildfire-2026`
- **Logged per run:** model parameters, per-horizon Brier/AUC/LogLoss, mean Brier, feature list, serialized model artifact
- **Backend:** local file store (`mlruns/`)

## Feature Engineering Hypotheses

1. **Interaction: distance × closing_speed** — fires that are close AND moving fast are the highest risk
2. **Ratio: projected_advance / dist_min** — fraction of distance covered per unit time
3. **Binary flags:** is_closing (closing_speed > 0), is_growing (area_growth > 0)
4. **Binned distance:** categorical distance bands (near/medium/far)
5. **Drop redundant features:** remove one from each highly correlated pair (r > 0.95)

## Risks and Unknowns

- **Small sample size (n=221)** is the primary risk — models may overfit easily
- **High censoring (68.8%)** means limited information about actual hit times
- **Feature sparsity** — many fires show zero movement, reducing signal for those observations
- **No geographic coordinates** — cannot use spatial modeling techniques
- **Submission format** requires cumulative probabilities at 4 time points — must be monotonically non-decreasing

## Phase 6 Decisions — Tuning, Ensembles, Error Analysis

### Hyperparameter tuning (Optuna, TPE sampler)

- Top 3 models from Phase 5 were tuned: `gradient_boosted_survival`, `random_survival_forest`, `xgboost`.
- Budget: 50 trials per model, 30 min timeout each.
- Search spaces defined in `configs/model_config.yaml → tuning.search_spaces`.
- Objective: minimize mean CV Brier across 5 stratified folds.
- All trial parameters/values logged to MLflow and `models/tuned_params.json`.

### Ensemble methods

Three ensembles built on the top-3 members (using tuned params when available):

1. **Weighted average** — convex combination whose weights minimize mean CV Brier on OOF predictions (`scipy.optimize.SLSQP`).
2. **Stacking** — per-horizon logistic-regression meta-learner trained on OOF base probabilities.
3. **Blending** — same meta-learner shape, but base models are trained on a 75% slice and the meta-learner is fit on the remaining 25% holdout.

All ensembles enforce monotonicity (P_12h ≤ P_24h ≤ P_48h ≤ P_72h) and clip to [0, 1].

### Final model selection rule

Lowest **OOF mean Brier** wins among:
- The 3 ensemble variants above.
- Each individual tuned member (in case a single model already beats the convex hull).

Winner is written to `models/phase6_best.txt`; `models/phase5_best_model.txt` is updated so `make submit` picks up the Phase 6 winner without code changes.

### Phase 6 Error Analysis

`notebooks/05_error_analysis.ipynb` surfaces three systematic weaknesses on OOF predictions:

1. **Distance-band concentration** — mid/long-distance fires (3rd–4th quartile of `dist_min_ci_0_5h`) contribute disproportionately to the total Brier. The kinetic features (`closing_speed`, `alignment_abs`) are noisier in that regime.
2. **Rare-event horizons** — the 12h horizon has very few positive events in CV folds; calibration curves show the biggest deviation there. Any monotonicity-preserving smoothing that lifts 12h toward 24h helps.
3. **Static fires** — samples where `is_closing=0`, `is_growing=0`, and `closing_speed≈0` are the hardest: the model has no kinetic signal, so predictions default to near the marginal event rate (~0.3), which scores poorly when the fire does hit late in the 72h window.

These patterns inform Phase 7–8 observability checks (drift detection on kinetic features, calibration monitoring per horizon).
