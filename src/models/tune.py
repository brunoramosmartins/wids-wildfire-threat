"""Hyperparameter tuning with Optuna (Phase 6).

Tunes the top Phase 5 models by minimizing mean CV Brier score across
4 horizons. Each trial runs stratified K-fold CV. Best hyperparameters
are saved to ``models/tuned_params.json`` and logged to MLflow.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

from src.models.boosting import get_xgboost_model
from src.models.evaluate import compute_metrics
from src.models.survival import get_gbs_model, get_rsf_model
from src.models.train import HorizonPredictor, _get_feature_and_target
from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)

# Silence Optuna's INFO logs (we use structlog)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_brier_mean(
    factory: Callable[[], HorizonPredictor],
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> float:
    """Mean Brier across folds — objective for Optuna (minimize)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    brier_list: list[float] = []
    for train_idx, val_idx in skf.split(X, y["event"]):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = factory()
        model.fit(X_tr, y_tr)
        preds = model.predict_proba_horizons(X_val)
        metrics = compute_metrics(y_val, preds)
        brier_list.append(metrics["brier_mean"])
    return float(np.mean(brier_list))


# --- Per-model Optuna objectives ---


def _suggest_gbs(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "min_samples_split": trial.suggest_int("min_samples_split", *space["min_samples_split"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *space["min_samples_leaf"]),
    }


def _suggest_rsf(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "min_samples_split": trial.suggest_int("min_samples_split", *space["min_samples_split"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *space["min_samples_leaf"]),
    }


def _suggest_xgb(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
    }


_SUGGESTERS: dict[str, Callable[[optuna.Trial, dict[str, Any]], dict[str, Any]]] = {
    "gradient_boosted_survival": _suggest_gbs,
    "random_survival_forest": _suggest_rsf,
    "xgboost": _suggest_xgb,
}

_FACTORIES: dict[str, Callable[..., HorizonPredictor]] = {
    "gradient_boosted_survival": get_gbs_model,
    "random_survival_forest": get_rsf_model,
    "xgboost": get_xgboost_model,
}


def _build_objective(
    model_name: str,
    space: dict[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> Callable[[optuna.Trial], float]:
    """Factory: returns Optuna objective fn for a given model."""
    suggest = _SUGGESTERS[model_name]
    factory_fn = _FACTORIES[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        params["random_state"] = seed

        def model_factory() -> HorizonPredictor:
            return factory_fn(**params)

        try:
            brier = _cv_brier_mean(model_factory, X, y, n_splits, seed)
        except Exception as e:
            logger.warning("trial_failed", model=model_name, error=str(e)[:200])
            raise optuna.TrialPruned() from e
        trial.set_user_attr("brier_mean_cv", brier)
        return brier

    return objective


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_trials: int,
    timeout_s: int,
    n_splits: int,
    seed: int,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Run Optuna study for a single model; log all trials to MLflow."""
    if model_name not in _SUGGESTERS:
        raise ValueError(f"Unknown model for tuning: {model_name}")

    logger.info(
        "tuning_start",
        model=model_name,
        n_trials=n_trials,
        timeout_s=timeout_s,
    )

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    config = load_config(Path("configs/model_config.yaml"))
    experiment_name = config["experiment_name"]
    mlflow.set_experiment(experiment_name)

    objective = _build_objective(model_name, search_space, X, y, n_splits, seed)

    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_s,
        show_progress_bar=False,
        catch=(Exception,),
    )
    elapsed = time.perf_counter() - t0

    # Did any trial succeed?
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.error(
            "tuning_no_completed_trials",
            model=model_name,
            n_trials_attempted=len(study.trials),
            seconds=round(elapsed, 1),
        )
        return {
            "model": model_name,
            "best_value": None,
            "best_params": {},
            "n_trials": len(study.trials),
            "n_completed": 0,
            "seconds": round(elapsed, 1),
            "error": "all trials failed",
        }

    best_value = study.best_value
    best_params = study.best_params

    # Log parent MLflow run with best params
    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        mlflow.log_params(
            {
                "type": model_name,
                "tuning": "optuna_tpe",
                "n_trials": n_trials,
                "n_splits": n_splits,
                "seed": seed,
            }
        )
        mlflow.log_metric("best_brier_mean_cv", best_value)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        trials_df = study.trials_dataframe()
        trials_path = Path(f"mlruns_trials_{model_name}.csv")
        trials_df.to_csv(trials_path, index=False)
        mlflow.log_artifact(str(trials_path))
        trials_path.unlink(missing_ok=True)

    logger.info(
        "tuning_complete",
        model=model_name,
        best_brier=round(best_value, 5),
        best_params=best_params,
        n_trials_done=len(study.trials),
        n_completed=len(completed),
        seconds=round(elapsed, 1),
    )

    return {
        "model": model_name,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "seconds": round(elapsed, 1),
    }


def run_tuning() -> dict[str, dict[str, Any]]:
    """Run Optuna on all models listed in ``configs/model_config.yaml → tuning.models``."""
    config = load_config(Path("configs/model_config.yaml"))
    data_config = load_config(Path("configs/data_config.yaml"))

    seed = int(config["random_seed"])
    n_splits = int(config["validation"]["n_splits"])
    set_seed(seed)

    tuning_cf = config.get("tuning", {})
    n_trials = int(tuning_cf.get("n_trials", 50))
    timeout_s = int(tuning_cf.get("timeout_seconds", 1800))
    models = list(tuning_cf.get("models", []))
    spaces = tuning_cf.get("search_spaces", {})

    feat_path = Path(data_config["paths"]["features"]) / data_config["feature_files"]["train"]
    df = read_parquet(feat_path)
    X, y, feature_cols = _get_feature_and_target(df)

    logger.info(
        "tuning_pipeline_start",
        models=models,
        n_trials=n_trials,
        samples=len(X),
        features=len(feature_cols),
    )

    all_results: dict[str, dict[str, Any]] = {}
    for model_name in models:
        if model_name not in spaces:
            logger.warning("no_search_space", model=model_name)
            continue
        all_results[model_name] = tune_model(
            model_name=model_name,
            X=X,
            y=y,
            n_trials=n_trials,
            timeout_s=timeout_s,
            n_splits=n_splits,
            seed=seed,
            search_space=spaces[model_name],
        )

    # Save best params
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tuned_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("tuned_params_saved", path=str(out_path))

    # Compare tuned vs Phase 5 baselines
    adv_path = out_dir / "advanced_results.json"
    if adv_path.is_file():
        adv = json.loads(adv_path.read_text(encoding="utf-8"))
        for m, r in all_results.items():
            baseline_brier = adv.get(m, {}).get("brier_mean_cv")
            if baseline_brier is not None:
                delta = r["best_value"] - baseline_brier
                logger.info(
                    "tuning_delta",
                    model=m,
                    baseline=round(baseline_brier, 5),
                    tuned=round(r["best_value"], 5),
                    delta=round(delta, 5),
                )

    return all_results


def main() -> None:
    logger.info("tune_cli_start")
    run_tuning()
    logger.info("tune_cli_complete")


if __name__ == "__main__":
    main()
