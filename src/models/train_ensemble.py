"""Ensemble training orchestrator (Phase 6).

Generates OOF predictions for the top-3 models using (optionally) tuned
hyperparameters, then trains and compares 3 ensemble strategies:

1. Weighted average (weights optimized on OOF)
2. Stacking (per-horizon logistic meta-learner on OOF)
3. Blending (single-holdout meta-learner)

Saves artifacts:
- ``models/oof_predictions_{model}.parquet`` — per-model OOF preds
- ``models/test_predictions_{model}.parquet`` — per-model test preds (from full-data refit)
- ``models/ensemble_results.json`` — CV metrics for each ensemble
- ``models/ensemble_weights.json`` — weighted-avg weights
- ``data/predictions/predictions_ensemble.parquet`` — best ensemble's test preds
- ``models/phase6_best.txt`` — name of best ensemble (or individual if better)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models.aft import get_lognormal_aft_model, get_weibull_aft_model
from src.models.baselines import HORIZONS
from src.models.boosting import get_xgboost_model
from src.models.calibration import IsotonicHorizonCalibrator
from src.models.ensemble import (
    BlendingEnsemble,
    StackingEnsemble,
    enforce_monotonicity,
    optimize_weights,
    weighted_average,
)
from src.models.evaluate import compute_metrics, evaluate_model
from src.models.seed_ensemble import SeedEnsembleWrapper
from src.models.survival import get_gbs_model, get_rsf_model
from src.models.train import HorizonPredictor, _get_feature_and_target
from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)

PROB_COLS = [f"prob_{h}h" for h in HORIZONS]


def _get_tabpfn_factory() -> Callable[..., HorizonPredictor] | None:
    """Lazy import — tabpfn is in the optional [advanced] extra."""
    try:
        from src.models.tabpfn_wrapper import get_tabpfn_model

        return get_tabpfn_model
    except ImportError:
        return None


_DEFAULT_FACTORIES: dict[str, Callable[..., HorizonPredictor]] = {
    "gradient_boosted_survival": get_gbs_model,
    "random_survival_forest": get_rsf_model,
    "xgboost": get_xgboost_model,
    "weibull_aft": get_weibull_aft_model,
    "lognormal_aft": get_lognormal_aft_model,
}

_tabpfn = _get_tabpfn_factory()
if _tabpfn is not None:
    _DEFAULT_FACTORIES["tabpfn"] = _tabpfn


def _resolve_factory(
    model_name: str,
    tuned_params: dict[str, Any] | None,
    model_defaults: dict[str, Any],
    seed: int,
    seed_ensemble_cfg: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
) -> Callable[[], HorizonPredictor]:
    """Build a zero-arg factory for ``model_name`` using tuned > defaults.

    If ``seed_ensemble_cfg['enabled']`` is truthy, the returned factory
    wraps the base model in a ``SeedEnsembleWrapper`` with the configured
    seeds. AFT/TabPFN are not wrapped (expensive or deterministic).

    If ``feature_names`` is provided and the model is XGBoost / LightGBM,
    the factory injects ``feature_names`` so monotone constraints kick in.
    """
    base_fn = _DEFAULT_FACTORIES[model_name]
    if tuned_params and model_name in tuned_params:
        params = dict(tuned_params[model_name].get("best_params", {}))
    else:
        params = dict(model_defaults.get(model_name, {}))
    # Stochastic models accept random_state; parametric AFTs do not.
    _STOCHASTIC = {
        "gradient_boosted_survival",
        "random_survival_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "tabpfn",
    }
    if model_name in _STOCHASTIC:
        params.setdefault("random_state", seed)
    if feature_names is not None and model_name in {"xgboost", "lightgbm"}:
        params.setdefault("feature_names", feature_names)

    def _base_factory() -> HorizonPredictor:
        return base_fn(**params)

    # Wrap in SeedEnsemble for models that benefit from seed-averaging
    # (exclude AFT — deterministic — and TabPFN — expensive).
    wrappable = {"gradient_boosted_survival", "random_survival_forest", "xgboost"}
    if seed_ensemble_cfg and seed_ensemble_cfg.get("enabled") and model_name in wrappable:
        seeds = list(seed_ensemble_cfg.get("seeds", [42, 123, 2024, 7, 99]))

        def _seeded_factory() -> HorizonPredictor:
            return SeedEnsembleWrapper(_base_factory, seeds=seeds)

        return _seeded_factory
    return _base_factory


def _oof_predictions(
    factory: Callable[[], HorizonPredictor],
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified K-fold OOF probability predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.DataFrame(index=X.index, columns=PROB_COLS, dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y["event"]), start=1):
        model = factory()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict_proba_horizons(X.iloc[va_idx])
        oof.iloc[va_idx] = preds[PROB_COLS].values
        logger.info("oof_fold_done", fold=fold)
    # Guard against any NaN
    return oof.fillna(0.5).astype(float)


def _full_refit_and_predict(
    factory: Callable[[], HorizonPredictor],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[HorizonPredictor, pd.DataFrame]:
    model = factory()
    model.fit(X_train, y_train)
    preds = model.predict_proba_horizons(X_test)
    return model, preds


def _holdout_predictions(
    factory: Callable[[], HorizonPredictor],
    X: pd.DataFrame,
    y: pd.DataFrame,
    holdout_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on train slice, predict on holdout slice; return (holdout_preds, holdout_y)."""
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X,
        y,
        test_size=holdout_size,
        stratify=y["event"],
        random_state=seed,
    )
    model = factory()
    model.fit(X_tr, y_tr)
    preds = model.predict_proba_horizons(X_ho)
    preds.index = X_ho.index
    return preds, y_ho


def run_ensembling() -> dict[str, Any]:
    config = load_config(Path("configs/model_config.yaml"))
    data_config = load_config(Path("configs/data_config.yaml"))

    seed = int(config["random_seed"])
    n_splits = int(config["validation"]["n_splits"])
    set_seed(seed)

    ensemble_cf = config.get("ensemble", {})
    members: list[str] = list(ensemble_cf.get("members", []))
    if not members:
        raise ValueError("No ensemble members in configs/model_config.yaml")

    model_defaults = config.get("models", {})
    seed_ensemble_cfg = ensemble_cf.get("seed_ensemble", {})
    calibrate_output = bool(ensemble_cf.get("calibrate_output", False))

    # Load tuned params (if available)
    tuned_path = Path("models/tuned_params.json")
    tuned_params: dict[str, Any] | None = None
    if tuned_path.is_file():
        tuned_params = json.loads(tuned_path.read_text(encoding="utf-8"))
        logger.info("using_tuned_params", models=list(tuned_params.keys()))
    else:
        logger.info("no_tuned_params_using_defaults")

    # Load train + test features
    feat_dir = Path(data_config["paths"]["features"])
    train_path = feat_dir / data_config["feature_files"]["train"]
    test_path = feat_dir / data_config["feature_files"]["test"]
    df_train = read_parquet(train_path)
    df_test = read_parquet(test_path)

    X, y, feature_cols = _get_feature_and_target(df_train)
    id_col = data_config["id_column"]
    X_test = df_test[feature_cols]
    test_ids = df_test[id_col].values

    logger.info(
        "ensemble_start",
        members=members,
        n_train=len(X),
        n_test=len(X_test),
        n_features=len(feature_cols),
    )

    # Step 1: OOF + test predictions per member
    oof_preds: dict[str, pd.DataFrame] = {}
    test_preds: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, HorizonPredictor] = {}

    out_models = Path("models")
    out_models.mkdir(parents=True, exist_ok=True)

    for member in members:
        if member not in _DEFAULT_FACTORIES:
            logger.warning("unknown_member_skipped", model=member)
            continue
        factory = _resolve_factory(
            member,
            tuned_params,
            model_defaults,
            seed,
            seed_ensemble_cfg,
            feature_names=feature_cols,
        )
        logger.info("ensemble_member_oof", model=member)
        oof = _oof_predictions(factory, X, y, n_splits, seed)
        oof_preds[member] = oof
        write_parquet(oof, out_models / f"oof_predictions_{member}.parquet")

        # Full-data refit → test predictions
        model, preds = _full_refit_and_predict(factory, X, y, X_test)
        test_preds[member] = preds
        fitted_models[member] = model
        write_parquet(preds, out_models / f"test_predictions_{member}.parquet")
        logger.info(
            "ensemble_member_ready",
            model=member,
            oof_brier=round(compute_metrics(y, oof)["brier_mean"], 5),
        )

    member_names = list(oof_preds.keys())
    oof_list = [oof_preds[m] for m in member_names]
    test_list = [test_preds[m] for m in member_names]

    # Step 2: weighted-average ensemble (weights optimized to MAXIMIZE Hybrid)
    logger.info("optimizing_weights")
    weights = optimize_weights(oof_list, y, seed=seed, objective="hybrid")
    weights_dict = dict(zip(member_names, weights.tolist(), strict=True))
    wa_oof = weighted_average(oof_list, weights.tolist())
    wa_test = weighted_average(test_list, weights.tolist())

    wa_metrics = compute_metrics(y, wa_oof)
    logger.info(
        "weighted_average_done",
        weights={k: round(v, 4) for k, v in weights_dict.items()},
        hybrid_oof=round(wa_metrics["hybrid_score"], 5),
        weighted_brier_oof=round(wa_metrics["weighted_brier"], 5),
    )

    # Step 3: stacking ensemble
    logger.info("fitting_stacking")
    stack = StackingEnsemble(random_state=seed, C=1.0).fit(oof_list, y)
    stack_oof = stack.predict_proba_horizons(oof_list)
    stack_test = stack.predict_proba_horizons(test_list)
    stack_metrics = compute_metrics(y, stack_oof)
    logger.info("stacking_done", hybrid_oof=round(stack_metrics["hybrid_score"], 5))

    # Step 4: blending (single holdout)
    logger.info("fitting_blending")
    holdout_base: list[pd.DataFrame] = []
    holdout_y: pd.DataFrame | None = None
    for member in member_names:
        factory = _resolve_factory(
            member,
            tuned_params,
            model_defaults,
            seed,
            seed_ensemble_cfg,
            feature_names=feature_cols,
        )
        ho_preds, ho_y = _holdout_predictions(factory, X, y, holdout_size=0.25, seed=seed)
        holdout_base.append(ho_preds)
        holdout_y = ho_y
    assert holdout_y is not None
    blend = BlendingEnsemble(random_state=seed, C=1.0).fit(holdout_base, holdout_y)
    blend_oof = blend.predict_proba_horizons(oof_list)
    blend_test = blend.predict_proba_horizons(test_list)
    blend_metrics = compute_metrics(y, blend_oof)
    logger.info("blending_done", hybrid_oof=round(blend_metrics["hybrid_score"], 5))

    # Step 5: compare — include per-member OOF Hybrid
    member_metrics = {m: compute_metrics(y, oof_preds[m]) for m in member_names}
    results: dict[str, Any] = {
        "members": {
            m: {
                "oof_hybrid_score": float(member_metrics[m]["hybrid_score"]),
                "oof_weighted_brier": float(member_metrics[m]["weighted_brier"]),
                "oof_c_index": float(member_metrics[m]["c_index"]),
            }
            for m in member_names
        },
        "weighted_average": {
            "oof_hybrid_score": wa_metrics["hybrid_score"],
            "weights": weights_dict,
            "metrics": wa_metrics,
        },
        "stacking": {
            "oof_hybrid_score": stack_metrics["hybrid_score"],
            "metrics": stack_metrics,
        },
        "blending": {
            "oof_hybrid_score": blend_metrics["hybrid_score"],
            "metrics": blend_metrics,
        },
    }

    # Step 6: log each ensemble to MLflow
    experiment_name = config["experiment_name"]
    mlflow.set_experiment(experiment_name)
    ensemble_oof = {
        "weighted_average": wa_oof,
        "stacking": stack_oof,
        "blending": blend_oof,
    }
    for name, oof_df in ensemble_oof.items():
        evaluate_model(
            model_name=f"ensemble_{name}",
            y_true=y,
            y_pred=oof_df,
            params={
                "type": f"ensemble_{name}",
                "members": member_names,
                "n_splits": n_splits,
                "seed": seed,
                "phase": 6.5,
                "objective": "hybrid",
            },
            feature_list=feature_cols,
            model_artifact=None,
        )

    # Step 7: pick best (HIGHEST OOF Hybrid) among ensembles AND individual members
    candidates: dict[str, tuple[float, pd.DataFrame]] = {
        "weighted_average": (wa_metrics["hybrid_score"], wa_test),
        "stacking": (stack_metrics["hybrid_score"], stack_test),
        "blending": (blend_metrics["hybrid_score"], blend_test),
    }
    for m in member_names:
        candidates[m] = (member_metrics[m]["hybrid_score"], test_preds[m])

    best_name, (best_hybrid, best_test_preds) = max(candidates.items(), key=lambda kv: kv[1][0])
    logger.info(
        "best_phase6",
        model=best_name,
        oof_hybrid=round(best_hybrid, 5),
    )
    results["best"] = {
        "model": best_name,
        "oof_hybrid_score": best_hybrid,
    }

    # Step 7.5: OPTIONAL isotonic calibration of best ensemble's OOF -> test
    calibration_applied = False
    if calibrate_output:
        # Map best name back to its OOF predictions (for ensembles, recompute)
        best_oof_map = {
            "weighted_average": wa_oof,
            "stacking": stack_oof,
            "blending": blend_oof,
            **{m: oof_preds[m] for m in member_names},
        }
        best_oof = best_oof_map[best_name]
        calib = IsotonicHorizonCalibrator().fit(best_oof, y)
        calibrated_test = calib.transform(best_test_preds)

        calib_oof = calib.transform(best_oof)
        calib_metrics = compute_metrics(y, calib_oof)
        logger.info(
            "calibration_applied",
            best=best_name,
            hybrid_before=round(best_hybrid, 5),
            hybrid_after=round(calib_metrics["hybrid_score"], 5),
            summary=calib.summary(),
        )
        # Accept calibration only if it improves OOF Hybrid
        if calib_metrics["hybrid_score"] >= best_hybrid:
            best_test_preds = calibrated_test
            best_hybrid = calib_metrics["hybrid_score"]
            calibration_applied = True
        else:
            logger.info("calibration_rejected_noimprovement")

    results["calibration_applied"] = calibration_applied

    # Save best ensemble test predictions under "ensemble" name for submission
    pred_dir = Path(data_config["paths"]["predictions"])
    pred_dir.mkdir(parents=True, exist_ok=True)

    ensemble_test = best_test_preds.copy()
    ensemble_test = enforce_monotonicity(ensemble_test)
    ensemble_test.insert(0, id_col, test_ids)
    write_parquet(ensemble_test, pred_dir / "predictions_ensemble.parquet")
    write_parquet(ensemble_test, pred_dir / f"predictions_{best_name}.parquet")

    # Persist artifacts
    (out_models / "ensemble_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    (out_models / "ensemble_weights.json").write_text(
        json.dumps(weights_dict, indent=2), encoding="utf-8"
    )
    (out_models / "phase6_best.txt").write_text(best_name + "\n", encoding="utf-8")
    # Also update the submission selector so `make submit` picks Phase 6 winner
    (out_models / "phase5_best_model.txt").write_text(
        ("ensemble" if best_name in {"weighted_average", "stacking", "blending"} else best_name)
        + "\n",
        encoding="utf-8",
    )

    return results


def main() -> None:
    logger.info("train_ensemble_cli_start")
    run_ensembling()
    logger.info("train_ensemble_cli_complete")


if __name__ == "__main__":
    main()
