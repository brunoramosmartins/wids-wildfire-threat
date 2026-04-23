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

---

## Phase 6.5 Decisions — Metric alignment + robustness

### Competition metric realization (Phase 6.5 origin)

Phase 6 was optimizing **mean Brier across 4 horizons (naive)**. The competition page (re-read after Phase 6 regression) specifies:

```
Hybrid = 0.3 × C-index + 0.7 × (1 − Weighted Brier)
Weighted Brier = 0.3 × Brier@24h + 0.4 × Brier@48h + 0.3 × Brier@72h  (censor-aware)
```

Three substantive differences from what we optimized in Phase 6:

1. **30% of the score is ranking** (C-index) — a pure Brier optimizer ignores this.
2. **48h has 40% weight**, 24h and 72h 30% each; **12h is NOT in the score** (but still required in submission format).
3. **Censor-aware Brier excludes** fires censored before the horizon — including them as 0 (what naive Brier does) biases toward under-prediction.

This likely explains the Phase 6 LB regression: we moved probabilities to minimize naive Brier in regions where the censor-aware metric didn't reward us.

### Phase 6.5 additions

| Component | Module | Role |
|-----------|--------|------|
| Official metric | `src/models/evaluate.py` | `censor_aware_brier_at_horizon`, `weighted_brier_score`, `harrell_c_index`, `hybrid_score`; all reports anchored on it |
| Hybrid-objective tuning | `src/models/tune.py` | Optuna maximizes Hybrid across 5-fold CV (was: minimize naive Brier) |
| Hybrid-objective ensembling | `src/models/ensemble.py` | `optimize_weights(..., objective='hybrid')` via `scipy.optimize.SLSQP` |
| Monotone constraints | `src/models/boosting.py` | Known physical priors (distance ↓, closing_speed ↑, alignment ↑, …) |
| Seed ensembling | `src/models/seed_ensemble.py` | Wraps trees/boosters in 5-seed averaging |
| Isotonic calibration | `src/models/calibration.py` | Per-horizon isotonic on OOF → test; applied only if it improves OOF Hybrid |
| Weibull / LogNormal AFT | `src/models/aft.py` | Parametric survival members for diversity |
| TabPFN wrapper | `src/models/tabpfn_wrapper.py` | Optional foundation-model member; pulls `[advanced]` extra |
| Repeated 5×10 K-fold | `src/validation/repeated_cv.py` | √10 variance reduction on OOF estimates |
| Nested CV | `src/validation/nested_cv.py` | Honest tuning estimate |
| Adversarial validation | `src/validation/adversarial.py` | Covariate-shift detector train↔test |

### Phase 6.5 validation strategy

- **CV:** stratified 5-fold on `event`, repeated 10× for reporting only (not during tuning; tuning uses 5-fold for speed).
- **Ensemble weight optimization:** on **OOF** Hybrid Score via SLSQP with simplex constraint.
- **Calibration:** fit isotonic on OOF, apply to test predictions, but only **accept if OOF Hybrid does not regress** (guards against overfitting the calibrator on a tiny dataset).

### Adversarial validation result

- AUC train-vs-test = **0.41 ± 0.06** → **no meaningful shift**.
- Implication: Phase 6's LB regression was not due to covariate shift. It was due to optimizing the wrong metric. Phase 6.5 fixes that directly.
