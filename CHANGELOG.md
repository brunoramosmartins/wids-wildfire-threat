# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Phase 5 — Advanced modeling
- Survival: Cox PH (lifelines), RSF e gradient boosting survival (scikit-survival) com `predict_proba_horizons` alinhado ao formato Kaggle
- Boosting: XGBoost, LightGBM e CatBoost (multi-horizon; mesmo contrato que baselines)
- Avaliação: C-index (Harrell) e gap de calibração em 72h; relatórios MLflow
- Seleção de features: RFE, permutation importance, comparação top-N (20/30/50) + CLI `src.features.selection`
- Orquestração: `train_advanced`, `Makefile` (`train-advanced`, `features-select`, `pipeline-advanced`), `scripts/run_pipeline_advanced.sh`
- Predição/submissão: `models/phase5_best_model.txt` como default quando existir
- Documentação: `docs/experiment_log.md` (Fase 5 + LB), `docs/feature_catalog.md` (seleção)
- Dependência: catboost

### Phase 0 — Setup & Alignment
- Initialize repository structure
- Set up Python environment, Makefile, and pre-commit
- Configure CI pipeline
- Add issue and PR templates
