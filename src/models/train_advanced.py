"""Train survival + gradient boosting models (Phase 5).

Logs each variant to MLflow, writes ``models/advanced_results.json``,
and ``models/phase5_best_model.txt`` (best by mean CV Brier).
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.models.boosting import (
    get_catboost_model,
    get_lightgbm_model,
    get_xgboost_model,
)
from src.models.evaluate import compute_metrics, evaluate_model
from src.models.survival import get_cox_ph_model, get_gbs_model, get_rsf_model
from src.models.train import HorizonPredictor, _cross_validate, _get_feature_and_target
from src.observability.logger import setup_logger
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


def _build_model_factories(
    model_cf: dict[str, Any],
) -> dict[str, tuple[Callable[[], HorizonPredictor], dict[str, Any]]]:
    """Map model name -> (factory, mlflow params dict)."""
    xs = model_cf.get("xgboost", {})
    lg = model_cf.get("lightgbm", {})
    cb = model_cf.get("catboost", {})
    cp = model_cf.get("cox_ph", {})
    rs = model_cf.get("random_survival_forest", {})
    gb = model_cf.get("gradient_boosted_survival", {})

    factories: dict[str, tuple[Callable[[], HorizonPredictor], dict[str, Any]]] = {
        "cox_ph": (
            lambda: get_cox_ph_model(
                penalizer=float(cp.get("penalizer", 0.1)),
                l1_ratio=float(cp.get("l1_ratio", 0.0)),
            ),
            {"type": "cox_ph", **cp},
        ),
        "random_survival_forest": (
            lambda: get_rsf_model(
                n_estimators=int(rs.get("n_estimators", 100)),
                max_depth=rs.get("max_depth"),
                min_samples_split=int(rs.get("min_samples_split", 6)),
                min_samples_leaf=int(rs.get("min_samples_leaf", 3)),
            ),
            {"type": "random_survival_forest", **rs},
        ),
        "gradient_boosted_survival": (
            lambda: get_gbs_model(
                n_estimators=int(gb.get("n_estimators", 100)),
                learning_rate=float(gb.get("learning_rate", 0.1)),
                max_depth=int(gb.get("max_depth", 3)),
                subsample=float(gb.get("subsample", 1.0)),
            ),
            {"type": "gradient_boosted_survival", **gb},
        ),
        "xgboost": (
            lambda: get_xgboost_model(
                n_estimators=int(xs.get("n_estimators", 300)),
                max_depth=int(xs.get("max_depth", 6)),
                learning_rate=float(xs.get("learning_rate", 0.1)),
                subsample=float(xs.get("subsample", 0.8)),
                colsample_bytree=float(xs.get("colsample_bytree", 0.8)),
            ),
            {"type": "xgboost", **xs},
        ),
        "lightgbm": (
            lambda: get_lightgbm_model(
                n_estimators=int(lg.get("n_estimators", 300)),
                max_depth=int(lg.get("max_depth", -1)),
                learning_rate=float(lg.get("learning_rate", 0.1)),
                subsample=float(lg.get("subsample", 0.8)),
                colsample_bytree=float(lg.get("colsample_bytree", 0.8)),
            ),
            {"type": "lightgbm", **lg},
        ),
        "catboost": (
            lambda: get_catboost_model(
                iterations=int(cb.get("iterations", 300)),
                depth=int(cb.get("depth", 6)),
                learning_rate=float(cb.get("learning_rate", 0.1)),
                l2_leaf_reg=float(cb.get("l2_leaf_reg", 3.0)),
            ),
            {"type": "catboost", **cb},
        ),
    }
    return factories


def train_advanced_models() -> None:
    config = load_config(Path("configs/model_config.yaml"))
    seed = int(config["random_seed"])
    n_splits = int(config["validation"]["n_splits"])
    set_seed(seed)

    adv = config.get("advanced", {})
    enabled: list[str] = list(adv.get("models", []))
    if not enabled:
        enabled = [
            "cox_ph",
            "random_survival_forest",
            "gradient_boosted_survival",
            "xgboost",
            "lightgbm",
            "catboost",
        ]

    data_config = load_config(Path("configs/data_config.yaml"))
    feat_path = Path(data_config["paths"]["features"]) / data_config["feature_files"]["train"]
    df = read_parquet(feat_path)
    X, y, feature_cols = _get_feature_and_target(df)

    logger.info(
        "advanced_training_start",
        samples=len(X),
        features=len(feature_cols),
        models=enabled,
    )

    model_cf = config["models"]
    factories_all = _build_model_factories(model_cf)
    results: dict[str, dict[str, Any]] = {}

    for model_name in enabled:
        if model_name not in factories_all:
            logger.warning("unknown_model_skipped", model=model_name)
            continue
        factory, params = factories_all[model_name]
        t0 = time.perf_counter()
        try:
            cv_metrics = _cross_validate(factory, model_name, X, y, feature_cols, n_splits, seed)
            final = factory()
            final.fit(X, y)
            full_preds = final.predict_proba_horizons(X)
            full_metrics = compute_metrics(y, full_preds)

            evaluate_model(
                model_name=model_name,
                y_true=y,
                y_pred=full_preds,
                params={**params, "n_splits": n_splits, "seed": seed, "phase": 5},
                feature_list=feature_cols,
                model_artifact=final,
            )

            elapsed = time.perf_counter() - t0
            results[model_name] = {
                "seconds_train_log": round(elapsed, 2),
                "n_features": len(feature_cols),
                "brier_mean_cv": cv_metrics["brier_mean_mean"],
                "brier_std_cv": cv_metrics["brier_mean_std"],
                **{k: v for k, v in full_metrics.items() if isinstance(v, float)},
            }
            logger.info(
                "advanced_model_ok",
                model=model_name,
                brier_cv=round(cv_metrics["brier_mean_mean"], 5),
                seconds=round(elapsed, 1),
            )
        except Exception:
            logger.error(
                "advanced_model_failed",
                model=model_name,
                error=traceback.format_exc(),
            )
            results[model_name] = {"error": traceback.format_exc()}

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    path_json = out_dir / "advanced_results.json"
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("advanced_results_saved", path=str(path_json))

    ok = {
        k: v
        for k, v in results.items()
        if isinstance(v, dict) and "brier_mean_cv" in v and "error" not in v
    }
    if ok:
        best = min(ok.items(), key=lambda kv: kv[1]["brier_mean_cv"])
        best_path = out_dir / "phase5_best_model.txt"
        best_path.write_text(best[0] + "\n", encoding="utf-8")
        logger.info(
            "phase5_best",
            model=best[0],
            brier_cv=round(best[1]["brier_mean_cv"], 5),
            path=str(best_path),
        )


def main() -> None:
    logger.info("advanced_training_start_cli")
    train_advanced_models()
    logger.info("advanced_training_complete")


if __name__ == "__main__":
    main()
