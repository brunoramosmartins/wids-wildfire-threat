# Experiment Log

## Como preencher esta tabela

- **O arquivo CSV que você envia não traz “métrica” nenhuma** — só probabilidades por linha. O **Public Score** é calculado **pelo Kaggle** depois do upload; você copia esse número do site.
- **CV Score / métricas locais** vêm do **código**:
  - Baseline: `models/baseline_results.json`.
  - Fase 5: `models/advanced_results.json` + runs no MLflow.
- **Metric (coluna):** nome **exato** na aba **Evaluation** do Kaggle (atualizar em `docs/problem_statement.md` quando possível).

### Onde ver o Public Score no Kaggle

1. Competição: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26  
2. Histórico de **Submissions** / **Submit Predictions**.  
3. Copiar **Public Score** por arquivo enviado.

### Ficheiros locais por envio

- CSV: `submissions/submission_<modelo>_<data>.csv`  
- Predições: `data/predictions/predictions_<modelo>.parquet`  
- Melhor modelo Fase 5 (CV): `models/phase5_best_model.txt` → hoje: `gradient_boosted_survival`

---

## Summary Table

| # | Date | Model | Features | Metric | CV Score | LB Score | Notes |
|---|------|-------|----------|--------|----------|----------|-------|
| 1 | *(ajustar data do 1º envio)* | random_forest | 47 (`train_features`) | *(nome oficial no Kaggle)* | mean Brier CV: **0.0268 ± 0.0109** | **0.94801** | Baseline / Fase 4. |
| 2 | *(data do envio GBS)* | gradient_boosted_survival | 47 | *(igual col. Metric)* | mean Brier CV: **0.0282 ± 0.0149** | **0.96259** | Melhor CV em `advanced_results.json`; pred.: `predictions_gradient_boosted_survival.parquet`. vs #1: 0.96259 vs 0.94801 — confirmar se “maior” ou “menor” é melhor na competição. |

---

## Detailed Entries

### #1 — Baseline (Fase 4)

- **LB (público):** 0.94801  
- **CV:** `models/baseline_results.json` → `random_forest` → `brier_mean_cv` / `brier_std_cv`  
- **Ajustes manuais:** data do envio e nome da métrica da competição na tabela acima.

### #2 — Fase 5 (melhor modelo por CV: GBS)

- **Seleção:** menor `brier_mean_cv` entre modelos sem erro em `train_advanced` → **gradient_boosted_survival** (ver também `models/phase5_best_model.txt`).  
- **CV:** 0.028170555… mean, 0.014914075… std.  
- **Fit completo** (referência local; pode reflectir sobreajuste ao train): Brier médio **~0.0159**, **C-index ~0.963**, calibration_gap_72h **~0.098**.  
- **LB (público):** **0.96259** (baseline #1: 0.94801 — interpretar ganho/perda conforme a direção da métrica no Kaggle).  
- **Cox PH:** na corrida que gerou o JSON abaixo falhou (shape `predict_survival_function` lifelines); corrigido no código (`survival.py` — transpose). Voltar a correr `train_advanced` se quiser métricas Cox na tabela.

---

## Phase 5 — Model comparison (advanced ML)

Fonte: `models/advanced_results.json` (run local **bruno**). Tempos = `seconds_train_log` (treino full + log MLflow aprox.).

| Model | n_features | Train+log (s) | CV Brier (mean ± std) | Full Brier mean | C-index | Notes |
|-------|------------|---------------|------------------------|-----------------|---------|-------|
| cox_ph | — | — | — | — | — | Erro CV naquela corrida (shape predições); **re-treinar** após fix no repo. |
| random_survival_forest | 47 | 7.16 | 0.0292 ± 0.0128 | 0.0170 | 0.912 | **2.º melhor CV** — forte candidato a ensemble (Fase 6). |
| **gradient_boosted_survival** | 47 | 5.15 | **0.0282 ± 0.0149** | 0.0159 | 0.963 | **Melhor CV** → `phase5_best_model.txt`. |
| xgboost | 47 | 8.64 | 0.0325 ± 0.0177 | 0.00126 | 0.921 | Brier no train muito baixo vs CV → risco de **overfit**; usar sobretudo métrica CV. |
| lightgbm | 47 | 7.44 | 0.0366 ± 0.0202 | ~1e−9 | 0.929 | Idem — memorização aparente no full train. |
| catboost | 47 | 30.39 | 0.0331 ± 0.0171 | 1.1e−5 | 0.908 | CV entre XGB e LightGBM. |

### Top 2–3 para Fase 6 (ensemble)

Ordenar por **CV Brier** (mais fiável que full train nos boosters “por horizonte”):

1. **gradient_boosted_survival**  
2. **random_survival_forest**  
3. **xgboost** *(validar no CV / calibração; não confiar só no Brier no train)*  

Alternativa conservadora ao XGB na torre: **catboost** (3.º por CV se excluires XGB).

---

*Última atualização automática do log (tabelas Fase 5 + linha Summary #2): métricas copiadas do `advanced_results.json` partilhado; LB da entrada #2 fica pendente até novo envio no Kaggle.*
