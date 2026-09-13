"""LightGBM + XGBoost model factories."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb

DEFAULTS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 800,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "verbosity": -1,
    "n_jobs": -1,
}

XGB_DEFAULTS: dict[str, Any] = {
    "objective": "binary:logistic",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cpu",
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
    "verbosity": 0,
    "n_jobs": -1,
}


def lgbm_params(**overrides: Any) -> dict[str, Any]:
    """LightGBM defaults with caller overrides."""

    p = dict(DEFAULTS)
    p.update(overrides)
    return p


def make_model(seed: int = 42, **overrides: Any) -> lgb.LGBMClassifier:
    """Construct an LGBMClassifier for binary AUC."""

    p = lgbm_params(random_state=seed, **overrides)
    return lgb.LGBMClassifier(**p)


def xgb_params(**overrides: Any) -> dict[str, Any]:
    """XGBoost defaults with caller overrides (Deotte variants)."""

    p = dict(XGB_DEFAULTS)
    p.update(overrides)
    return p


def make_xgb_model(seed: int = 42, **overrides: Any) -> Any:
    """Construct an XGBClassifier (hist, CPU) for the Deotte blend."""

    import xgboost as xgb

    p = xgb_params(random_state=seed, **overrides)
    return xgb.XGBClassifier(**p)


CB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "AUC",
    "thread_count": -1,
    "verbose": False,
}

CB_FIT_KWARGS: dict[str, Any] = {
    "early_stopping_rounds": 50,
}


def cb_params(**overrides: Any) -> dict[str, Any]:
    """CatBoost defaults with caller overrides."""

    p = dict(CB_DEFAULTS)
    p.update(overrides)
    return p


def make_catboost_model(seed: int = 42, **overrides: Any) -> Any:
    """Construct a CatBoostClassifier for binary AUC."""

    from catboost import CatBoostClassifier

    p = cb_params(random_seed=seed, **overrides)
    return CatBoostClassifier(**p)


HGB_DEFAULTS: dict[str, Any] = {
    "max_iter": 50,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "early_stopping": False,
    "random_state": 42,
}


def hgb_params(**overrides: Any) -> dict[str, Any]:
    """HistGradientBoostingClassifier defaults with caller overrides."""

    p = dict(HGB_DEFAULTS)
    p.update(overrides)
    return p


def make_hgb_model(seed: int = 42, **overrides: Any) -> Any:
    """Construct a HistGradientBoostingClassifier for binary AUC."""

    from sklearn.ensemble import HistGradientBoostingClassifier

    p = hgb_params(random_state=seed, **overrides)
    return HistGradientBoostingClassifier(**p)


def short_params(p: dict[str, Any] | None = None) -> str:
    """One-line key params for EXPERIMENTS.md."""
    p = p or DEFAULTS
    return (
        f"n_estimators={p.get('n_estimators')} lr={p.get('learning_rate')} "
        f"num_leaves={p.get('num_leaves')}"
    )


def short_xgb_params(p: dict[str, Any] | None = None) -> str:
    """One-line XGB key params for EXPERIMENTS.md."""

    p = p or XGB_DEFAULTS
    return (
        f"n_estimators={p.get('n_estimators')} lr={p.get('learning_rate')} "
        f"max_depth={p.get('max_depth')}"
    )
