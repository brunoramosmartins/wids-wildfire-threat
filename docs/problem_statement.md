# Problem Statement

## Competition

- **Name:** WiDS Global Datathon 2026
- **URL:** https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26
- **Task:** Predicting Time-to-Threat for Evacuation Zones Using Survival Analysis

## Problem Description

When a wildfire ignites, emergency managers must decide which communities to warn, when to warn them, and where to position scarce resources — before certainty is available. The response requires both **prioritization** (which fires are most urgent soon) and **calibrated risk estimates** (how likely a fire is to threaten evacuation zones within actionable time windows).

The task is framed as survival analysis on real early-incident signals. Features are computed strictly from the first 5 hours after the initial perimeter observation (t0). The target is the time from `t0 + 5h` until the fire comes within 5 km of any evacuation-zone centroid.

The strongest solutions deliver two outcomes at once:
1. Rank fires correctly by urgency (for triage).
2. Produce calibrated probabilities (for threshold-based decisions).

## Target Variable

- `time_to_hit_hours` (float, 0–72): survival time from t0+5h.
- `event` (binary): 1 if the fire hit within 72 h, 0 if right-censored (never hit within the window).

Right-censoring distribution:
- **Event = 1 (hit):** `time_to_hit_hours` is the observed time.
- **Event = 0 (censored):** `time_to_hit_hours` is the last observed time in the window (≤ 72).

## Evaluation Metric

**Primary metric — Hybrid Score** (higher is better):

```
Hybrid = 0.3 × C-index  +  0.7 × (1 − Weighted Brier)
```

### Component 1 — C-index (30% weight)

Harrell's concordance index on ranked risk. Measures how well predictions **rank** fires by urgency. Range [0.5, 1.0]; higher is better. For a scalar risk we use `prob_72h` (the cumulative probability of hitting by 72 h).

### Component 2 — Weighted Brier (70% weight)

**Censor-aware** Brier score at 3 horizons (12h is NOT in the metric, but IS in the submission format):

```
Weighted Brier = 0.3 × Brier@24h + 0.4 × Brier@48h + 0.3 × Brier@72h
```

Per-horizon label rules:

| Status | Rule at horizon H |
|--------|-------------------|
| Hit within H (`event=1` AND `time_to_hit_hours ≤ H`) | label = 1 |
| Hit after H (`event=1` AND `time_to_hit_hours > H`) | label = 0 |
| Censored after H (`event=0` AND `time_to_hit_hours > H`) | label = 0 |
| **Censored before H (`event=0` AND `time_to_hit_hours ≤ H`)** | **EXCLUDED** |

The exclusion rule is critical: fires that leave the observation window before horizon H provide no information about what happens at H, and including them as 0 would bias the Brier toward models that underpredict.

### Horizon weighting rationale

- **48 h gets the highest weight (0.4)** — the sweet spot between lead time and decision urgency.
- 24 h and 72 h get 0.3 each.
- 12 h is required in the submission format but **not scored**.

## Submission Format

- **File:** CSV with columns `event_id, prob_12h, prob_24h, prob_48h, prob_72h`.
- **Rows:** 95 (one per test event), joined on `event_id`.
- **Values:** probabilities in [0, 1].
- **Monotonicity:** `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h` per row.

## Survival Analysis Framing

- Right-censored time-to-event.
- Observation window: 72 hours from t0+5h.
- Features are snapshots from the 0–5 h window only → no leakage of post-horizon information.

## Key Constraints

- **Deadline:** competition-specific (see Kaggle page).
- **External data:** permitted only per competition rules.
- **Model output:** cumulative P(hit by T) for T ∈ {12, 24, 48, 72}; must be monotonically non-decreasing in T.
