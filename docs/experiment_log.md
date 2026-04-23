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

---

## Phase 6.5 — Metric alignment + advanced techniques

Motivação: a Fase 6 otimizava **Brier mean (4 horizontes, sem censura)**. A métrica oficial da competição é:

```
Hybrid = 0.3 × C-index + 0.7 × (1 − Weighted Brier)
Weighted Brier = 0.3 × B@24h + 0.4 × B@48h + 0.3 × B@72h
```

…e a Brier é **censor-aware** (fires censurados antes do horizonte são EXCLUÍDOS, não somados como 0). Todo o tuning e seleção de modelo foram refeitos com essa métrica correta.

Novidades adicionadas nesta fase:

| Feature | Módulo | Motivação |
|---------|--------|-----------|
| Hybrid Score + censor-aware Brier | `src/models/evaluate.py` | Otimizar a métrica certa |
| Monotone constraints (XGB/LGBM) | `src/models/boosting.py` | Sinais físicos conhecidos (distância ↓, closing_speed ↑, alignment ↑) reduzem overfit |
| Isotonic calibration por horizonte | `src/models/calibration.py` | 70% da Hybrid é Brier → ganho direto se OOF mal calibrado |
| Seed ensembling wrapper | `src/models/seed_ensemble.py` | Média de 5 seeds reduz variância estocástica em n=221 |
| Weibull + LogNormal AFT | `src/models/aft.py` | Diversidade paramétrica vs boosters |
| TabPFN (opcional) | `src/models/tabpfn_wrapper.py` | Modelo de fundação para dados tabulares pequenos |
| Repeated 5×10 Stratified K-fold | `src/validation/repeated_cv.py` | √10 redução de variância na estimativa OOF |
| Nested CV helper | `src/validation/nested_cv.py` | Diagnóstico de honestidade do tuning |
| Adversarial validation | `src/validation/adversarial.py` | Detecta covariate shift train↔test |

### Adversarial validation train↔test

- **CV AUC:** 0.4132 ± 0.0619
- **Verdict:** "No meaningful shift — train/test look drawn from the same distribution."

**Conclusão:** não há covariate shift. Isso descarta sample-reweighting / subset selection como soluções rápidas. O gap OOF→LB do #3 era devido à métrica errada, não a drift de features.

### Tuning com Hybrid objective (Optuna TPE, 50 trials, 5-fold CV)

| Model | Fase 6 (Brier obj) Hybrid implícito | Fase 6.5 tuned Hybrid | Best params |
|-------|-------------------------------------|-----------------------|-------------|
| gradient_boosted_survival | — | **0.97352** | n_est=318, lr=0.049, depth=2, subsample=0.74, min_split=8, min_leaf=4 |
| random_survival_forest | — | 0.96130 | n_est=384, depth=6, min_split=4, min_leaf=1 |
| xgboost | — | 0.96909 | n_est=589, depth=6, lr=0.202, subsample=0.75, colsample=0.87, reg_lambda=0.51 |

### Ensembles Fase 6.5 (OOF Hybrid Score, com seed ensembling + monotone constraints)

| Candidato | OOF Hybrid | OOF Weighted Brier | OOF C-index |
|-----------|-----------|--------------------|-------------|
| **GBS tuned (5-seed ensemble)** | **0.97183** | ~0.018 | ~0.94 |
| XGBoost tuned (5-seed ensemble) | 0.95840 | — | — |
| RSF tuned (5-seed ensemble) | 0.96107 | — | — |
| Weibull AFT | 0.88xx | 0.073 | — |
| ensemble **weighted_average** | 0.96902 | 0.01456 | 0.93069 |
| ensemble stacking | 0.96641 | 0.01601 | 0.92539 |
| ensemble blending | 0.96068 | 0.02590 | 0.92936 |

**Observações:**

1. **GBS tuned sozinho venceu** — alinha com a suspeita do relatório de Fase 6 de que a mistura estava a puxar as probabilidades para regiões sub-ótimas.
2. **Weighted_average identificou Weibull AFT como membro inútil** (peso final = 0.0) — sinal de que a optimização SLSQP funciona; a diversidade paramétrica não trouxe valor aqui.
3. **Isotonic calibration foi REJEITADA** (OOF Hybrid caiu de 0.972 para 0.843 quando aplicada). Com só ~69 eventos e muitos sub-horizontes, o calibrador sobre-ajusta. Lógica de `calibrate_output=True ∧ only_if_improves_OOF` funcionou corretamente.
4. **Stacking meta-learner degradou vs weighted_average** (0.966 vs 0.969) — confirmando a intuição da Fase 6 de que com n=221 não há graus de liberdade suficientes para um meta-learner por horizonte.

### Entrada #4 — Phase 6.5 final submission

| # | Date | Model | Features | Metric | CV Score | LB Score | Notes |
|---|------|-------|----------|--------|----------|----------|-------|
| 4 | 2026-04-23 | gradient_boosted_survival (5-seed, tuned, Hybrid obj) | 47 | Hybrid | **0.97183** (OOF) | *(preencher)* | File: `submissions/submission_gradient_boosted_survival_2026-04-23.csv`. Phase 6.5 — métrica alinhada + seed ensembling + monotone constraints. Best params: n_est=318, lr=0.049, depth=2, min_split=8, min_leaf=4. |

### Summary Table — Phase 6.5 adicionada

| # | Date | Model | CV Score (official metric) | LB Score | Δ vs prev |
|---|------|-------|----------------------------|----------|-----------|
| 1 | *(Fase 4)* | random_forest | 0.0268 ± 0.0109 (Brier) | 0.94801 | — |
| 2 | *(Fase 5)* | gradient_boosted_survival | 0.0282 ± 0.0149 (Brier) | **0.96259** | +0.01458 |
| 3 | 2026-04-22 | ensemble weighted_average (Brier-tuned) | 0.02388 (OOF Brier) | 0.96190 | −0.00069 |
| 4 | 2026-04-23 | GBS (Hybrid-tuned, 5-seed, monotone) | **0.97183** (OOF Hybrid) | *(pendente)* | *(a medir)* |
- **Erros sistemáticos** documentados em `notebooks/05_error_analysis.ipynb`: concentração em faixas de distância médias/longas, baixa calibração no horizonte 12h, fires estáticos (is_closing=0, is_growing=0).
