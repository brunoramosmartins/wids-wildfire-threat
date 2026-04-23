# Adversarial Validation — Train vs. Test

Cross-validated AUC of a RandomForest distinguishing train from test rows.

- **CV AUC:** 0.4132 ± 0.0619
- **Interpretation:** No meaningful shift — train/test look drawn from the same distribution.

## Top 15 most discriminating features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `dist_min_log` | 0.0983 |
| 2 | `log1p_area_first` | 0.0926 |
| 3 | `dist_min_ci_0_5h` | 0.0845 |
| 4 | `area_first_ha` | 0.0822 |
| 5 | `hour_cos` | 0.0444 |
| 6 | `event_start_dayofweek` | 0.0426 |
| 7 | `hour_sin` | 0.0406 |
| 8 | `event_start_hour` | 0.0405 |
| 9 | `dist_accel_m_per_h2` | 0.0365 |
| 10 | `alignment_cos` | 0.0344 |
| 11 | `alignment_abs` | 0.0325 |
| 12 | `dist_slope_ci_0_5h` | 0.0304 |
| 13 | `dt_first_last_0_5h` | 0.0300 |
| 14 | `event_start_month` | 0.0295 |
| 15 | `month_cos` | 0.0269 |

Full ranking in `reports\data_quality\adversarial_feature_importance.csv`.