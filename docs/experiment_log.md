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

---

## Phase 6 — Hyperparameter tuning + ensembles

Fontes:
- `models/tuned_params.json` — best params por Optuna (50 trials, TPE sampler).
- `models/ensemble_results.json` — Brier OOF para `weighted_average`, `stacking`, `blending`.
- `models/ensemble_weights.json` — pesos optimizados (SLSQP) da média ponderada.
- `models/phase6_best.txt` — nome do vencedor Fase 6 (ensemble ou membro individual).

### Tuning (Optuna TPE, 50 trials/model, 5-fold CV)

| Model | Baseline CV Brier (Fase 5) | Tuned CV Brier | Δ | Best params |
|-------|----------------------------|----------------|---|-------------|
| gradient_boosted_survival | 0.02817 | **0.02445** | −0.00372 | n_est=293, lr=0.045, depth=2, subsample=0.998, min_split=9, min_leaf=5 |
| random_survival_forest | 0.02921 | **0.02893** | −0.00028 | n_est=169, depth=10, min_split=4, min_leaf=2 |
| xgboost | 0.03250 | **0.02789** | −0.00461 | n_est=148, depth=6, lr=0.039, subsample=0.62, colsample=0.69, reg_lambda=1.66 |

### Ensembles (OOF Brier on tuned members)

| Ensemble | OOF Brier mean | Notes |
|----------|----------------|-------|
| **weighted_average** | **0.02388** | Weights: GBS=0.713, RSF=0.188, XGB=0.099 (SLSQP on OOF) |
| stacking | 0.02448 | Per-horizon logistic meta-learner |
| blending | 0.03499 | 25% holdout; worse than members → insufficient signal with small data |
| GBS (tuned) alone | 0.02438 | Best single model |
| RSF (tuned) alone | 0.02885 | |
| XGB (tuned) alone | 0.02780 | |

Winner: **weighted_average** (0.02388, beats best single by ~0.00050). See `models/phase6_best.txt`.

### Entrada #3 — Phase 6 final submission

| # | Date | Model | Features | Metric | CV Score | LB Score | Notes |
|---|------|-------|----------|--------|----------|----------|-------|
| 3 | 2026-04-22 | ensemble (weighted_average) | 47 | *(nome oficial no Kaggle)* | mean Brier OOF: **0.02388** | **0.96190** | Fase 6: Optuna + weighted average of GBS(0.71)/RSF(0.19)/XGB(0.10). File: `submissions/submission_ensemble_2026-04-22.csv`. **LB ↓ vs #2 (0.96259 → 0.96190)** — ver notas. |

### Summary Table — atualização

| # | Date | Model | CV Score | LB Score | Δ vs prev |
|---|------|-------|----------|----------|-----------|
| 1 | *(Fase 4)* | random_forest | 0.0268 ± 0.0109 | 0.94801 | — |
| 2 | *(Fase 5)* | gradient_boosted_survival | 0.0282 ± 0.0149 | **0.96259** | +0.01458 |
| 3 | 2026-04-22 | ensemble weighted_average (tuned) | 0.02388 (OOF) | 0.96190 | **−0.00069** |

### Notas de Fase 6 — análise do regresso no LB

**CV melhorou ~15% mas o LB caiu ligeiramente** — sinal clássico de overfit ao OOF com dataset pequeno (n=221, 69% censurado). Interpretação:

1. **GBS sozinho (fase 5) foi calibrado no LB**; o tuning + mistura com RSF/XGB empurrou as probabilidades para regiões que o OOF favorecia mas o LB não reproduziu.
2. **Stacking Brier foi 0.02448** (quase igual ao weighted avg 0.02388) mas tem mais graus de liberdade por horizonte — o ganho aparente do weighted_average pode estar dentro do ruído do split CV.
3. **Blending = 0.03499** já sinalizava fragilidade: quando o meta-learner vê apenas 25% dos dados, o hold-out tem <16 eventos positivos — curva de learning muito ruidosa.

**Próximas ações sugeridas** (Fase 7–8):
- Submeter *GBS tuned sozinho* (sem ensemble) para isolar o efeito do tuning do efeito da mistura.
- Usar **repeated stratified K-fold (e.g. 5×5)** em vez de K-fold simples para estabilizar a estimativa OOF antes de optimizar pesos.
- **Calibração explícita** (Platt/isotonic por horizonte) antes da média ponderada — o `calibration_gap_72h=0.045` do weighted_average sugere que há margem.

### Notas de Fase 6 — gerais

- **Overfit check:** o weighted_average só melhorou ~0.0005 sobre o GBS tuned sozinho no OOF; esse ganho não se traduziu no LB, confirmando o risco de shrinkage insuficiente.
- **Erros sistemáticos** documentados em `notebooks/05_error_analysis.ipynb`: concentração em faixas de distância médias/longas, baixa calibração no horizonte 12h, fires estáticos (is_closing=0, is_growing=0).
