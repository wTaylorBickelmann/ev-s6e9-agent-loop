"""Stratified CV, persist OOF + fold models, append EXPERIMENTS.md.

Independent folds may run in a process pool. Per-worker `n_jobs` / CatBoost
`thread_count` are capped so workers do not oversubscribe the machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ev_s6e9.experiments import append_chunk, format_chunk
from ev_s6e9.features import FeatureBuilder, TE_COL, encode_target, split_xy, te_apply, te_fit, te_oof
from ev_s6e9.metrics import auc, fmt_cv, mean_std
from ev_s6e9.model import (
    CB_DEFAULTS,
    CB_FIT_KWARGS,
    DEFAULTS,
    HGB_DEFAULTS,
    make_catboost_model,
    make_hgb_model,
    make_model,
    short_params,
)
from ev_s6e9.parallel import cap_model_threads, map_jobs, resolve_max_workers
from ev_s6e9.paths import CV_JSON, OOF_CSV, OUTPUTS
from ev_s6e9.schema import ID_COL, TARGET


@dataclass
class CvResult:
    """LightGBM CV: OOF, fold AUCs, models, and mean importances."""

    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    models: list = field(default_factory=list)
    importances: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)


def _lgbm_one_fold(job: tuple) -> tuple:
    """Fit one LightGBM fold; return `(i, va, proba, auc, model, importances, cols)`."""

    df, y, fb, x_raw, use_deotte, tr, va, i, seed, early_stopping, overrides = job
    if use_deotte:
        x_tr = fb.transform(df.iloc[tr])
        x_va = fb.transform(df.iloc[va])
        if fb.te:
            v_tr = pd.to_numeric(df.iloc[tr][TE_COL], errors="coerce")
            v_va = pd.to_numeric(df.iloc[va][TE_COL], errors="coerce")
            mp, prior = te_fit(v_tr, y[tr])
            x_tr[TE_COL + "_te"] = te_oof(v_tr, y[tr], seed=seed)
            x_va[TE_COL + "_te"] = te_apply(v_va, mp, prior)
            mp5, prior5 = te_fit(v_tr, y[tr], m=5.0)
            x_tr[TE_COL + "_te_m5"] = te_oof(v_tr, y[tr], seed=seed, m=5.0)
            x_va[TE_COL + "_te_m5"] = te_apply(v_va, mp5, prior5)
            mp2, prior2 = te_fit(v_tr, y[tr], m=2.0)
            x_tr[TE_COL + "_te_m2"] = te_oof(v_tr, y[tr], seed=seed, m=2.0)
            x_va[TE_COL + "_te_m2"] = te_apply(v_va, mp2, prior2)
            c_tr = pd.to_numeric(df.iloc[tr]["Daily_Commute_km"], errors="coerce")
            c_va = pd.to_numeric(df.iloc[va]["Daily_Commute_km"], errors="coerce")
            cm, cp = te_fit(c_tr, y[tr])
            x_tr["Daily_Commute_km_te"] = te_oof(c_tr, y[tr], seed=seed)
            x_va["Daily_Commute_km_te"] = te_apply(c_va, cm, cp)
            cm5, cp5 = te_fit(c_tr, y[tr], m=5.0)
            x_tr["Daily_Commute_km_te_m5"] = te_oof(c_tr, y[tr], seed=seed, m=5.0)
            x_va["Daily_Commute_km_te_m5"] = te_apply(c_va, cm5, cp5)
            cm2, cp2 = te_fit(c_tr, y[tr], m=2.0)
            x_tr["Daily_Commute_km_te_m2"] = te_oof(c_tr, y[tr], seed=seed, m=2.0)
            x_va["Daily_Commute_km_te_m2"] = te_apply(c_va, cm2, cp2)
    else:
        x_tr = x_raw.iloc[tr]
        x_va = x_raw.iloc[va]
    m = make_model(seed=seed + i, **overrides)
    m.fit(
        x_tr,
        y[tr],
        eval_X=x_va,
        eval_y=y[va],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    p = m.predict_proba(x_va)[:, 1]
    return i, va, p, auc(y[va], p), m, m.feature_importances_, list(x_va.columns)


def run_cv(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    early_stopping: int = 50,
    model_overrides: dict | None = None,
    freq: bool = False,
    te: bool = False,
    test: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> CvResult:
    """Stratified LightGBM CV with early stopping; fill OOF and importances.

    When `freq` or `te` is set, features come from the Deotte FeatureBuilder
    (fold-safe target encoding, matching the `deotte` strategy); otherwise the
    raw `split_xy` representation is used.
    """
    y = encode_target(df[TARGET]).to_numpy()
    use_deotte = freq or te
    fb = FeatureBuilder(freq=freq, te=te).fit(df, test) if use_deotte else None
    x_raw = None if use_deotte else split_xy(df)[0]
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(skf.split(df, y))
    n_workers = resolve_max_workers(len(splits), max_workers)
    overrides = cap_model_threads(dict(model_overrides or {}), n_workers)
    jobs = [
        (df, y, fb, x_raw, use_deotte, tr, va, i, seed, early_stopping, overrides)
        for i, (tr, va) in enumerate(splits)
    ]
    rows = sorted(map_jobs(_lgbm_one_fold, jobs, n_workers), key=lambda r: r[0])
    models, scores, imps = [], [], []
    last_cols: list[str] = []
    for _i, va, p, score, m, imp, cols in rows:
        oof[va] = p
        scores.append(score)
        models.append(m)
        imps.append(imp)
        last_cols = cols
    mean, std = mean_std(scores)
    imp = pd.DataFrame(imps, columns=last_cols).mean(axis=0).sort_values(ascending=False)
    params = {**DEFAULTS, **(model_overrides or {})}
    return CvResult(oof, scores, mean, std, models, imp.to_frame("importance"), params)


def _catboost_one_fold(job: tuple) -> tuple:
    """Fit one CatBoost fold; return `(i, va, proba, auc, model, importances, cols)`."""

    df, y, fb, tr, va, i, seed, overrides = job
    x_tr = fb.transform(df.iloc[tr])
    x_va = fb.transform(df.iloc[va])
    if fb.te:
        v_tr = pd.to_numeric(df.iloc[tr][TE_COL], errors="coerce")
        v_va = pd.to_numeric(df.iloc[va][TE_COL], errors="coerce")
        mp, prior = te_fit(v_tr, y[tr])
        x_tr[TE_COL + "_te"] = te_oof(v_tr, y[tr], seed=seed)
        x_va[TE_COL + "_te"] = te_apply(v_va, mp, prior)
    m = make_catboost_model(seed=seed + i, **overrides)
    m.fit(x_tr, y[tr], eval_set=(x_va, y[va]), **CB_FIT_KWARGS)
    p = m.predict_proba(x_va)[:, 1]
    return i, va, p, auc(y[va], p), m, m.feature_importances_, list(x_va.columns)


def run_cv_catboost(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    model_overrides: dict | None = None,
    freq: bool = True,
    te: bool = True,
    test: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> CvResult:
    """Stratified CatBoost CV on Deotte features (freq + TE)."""
    y = encode_target(df[TARGET]).to_numpy()
    fb = FeatureBuilder(freq=freq, te=te).fit(df, test)
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(skf.split(df, y))
    n_workers = resolve_max_workers(len(splits), max_workers)
    overrides = cap_model_threads(dict(model_overrides or {}), n_workers, keys=("thread_count",))
    jobs = [(df, y, fb, tr, va, i, seed, overrides) for i, (tr, va) in enumerate(splits)]
    rows = sorted(map_jobs(_catboost_one_fold, jobs, n_workers), key=lambda r: r[0])
    models, scores, imps = [], [], []
    last_cols: list[str] = []
    for _i, va, p, score, m, imp, cols in rows:
        oof[va] = p
        scores.append(score)
        models.append(m)
        imps.append(imp)
        last_cols = cols
    mean, std = mean_std(scores)
    imp = pd.DataFrame(imps, columns=last_cols).mean(axis=0).sort_values(ascending=False)
    params = {**CB_DEFAULTS, **(model_overrides or {})}
    return CvResult(oof, scores, mean, std, models, imp.to_frame("importance"), params)


def save_run(df: pd.DataFrame, cv: CvResult, out: Path | None = None) -> None:
    """Write OOF, fold joblibs, feature_importance.csv, and cv.json."""

    out = out or OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    models = out / "models"
    models.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: df[ID_COL], TARGET: cv.oof}).to_csv(out / OOF_CSV.name, index=False)
    for i, m in enumerate(cv.models):
        joblib.dump(m, models / f"fold{i}.joblib")
    if cv.importances is not None:
        cv.importances.to_csv(out / "feature_importance.csv")
    payload = {
        "fold_aucs": cv.fold_aucs,
        "mean": cv.mean,
        "std": cv.std,
        "cv": fmt_cv(cv.mean, cv.std),
        "params": {k: v for k, v in cv.params.items() if k != "random_state"},
        "n": int(len(df)),
        "folds": len(cv.fold_aucs),
    }
    (out / CV_JSON.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_experiment(
    cv: CvResult,
    *,
    folds: int,
    note: str = "",
    path: Path | None = None,
) -> Path:
    """Append a LightGBM chunk to EXPERIMENTS.md."""

    cols = list(cv.importances.index) if cv.importances is not None else []
    feat = "Deotte freq+TE" if (any(c.endswith("_cnt") for c in cols) or "Annual_Income_USD_te" in cols) else "raw+charging_total"
    title = f"LightGBM {short_params(cv.params)}, {feat}, {folds}-fold"
    chunk = format_chunk(
        title,
        fmt_cv(cv.mean, cv.std),
        takeaway=note or "auto-logged from train",
    )
    return append_chunk(chunk, path=path)


def train(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    model_overrides: dict | None = None,
    freq: bool = False,
    te: bool = False,
    test: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> CvResult:
    """Run LightGBM CV, persist artifacts, optionally append EXPERIMENTS.md."""
    cv = run_cv(
        df,
        folds=folds,
        seed=seed,
        model_overrides=model_overrides,
        freq=freq,
        te=te,
        test=test,
        max_workers=max_workers,
    )
    save_run(df, cv, out=out)
    if log:
        log_experiment(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC: {fmt_cv(cv.mean, cv.std)}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv


def train_catboost(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    model_overrides: dict | None = None,
    freq: bool = True,
    te: bool = True,
    test: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> CvResult:
    """Run CatBoost CV on Deotte features, persist artifacts, optionally log."""
    cv = run_cv_catboost(
        df,
        folds=folds,
        seed=seed,
        model_overrides=model_overrides,
        freq=freq,
        te=te,
        test=test,
        max_workers=max_workers,
    )
    save_run(df, cv, out=out)
    if log:
        log_experiment(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC (CatBoost): {fmt_cv(cv.mean, cv.std)}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv


def _hgb_one_fold(job: tuple) -> tuple:
    """Fit one HGB fold; return `(i, va, proba, auc, model)`."""

    df, y, fb, tr, va, i, seed, overrides = job
    x_tr = fb.transform(df.iloc[tr])
    x_va = fb.transform(df.iloc[va])
    if fb.te:
        v_tr = pd.to_numeric(df.iloc[tr][TE_COL], errors="coerce")
        v_va = pd.to_numeric(df.iloc[va][TE_COL], errors="coerce")
        mp, prior = te_fit(v_tr, y[tr])
        x_tr[TE_COL + "_te"] = te_oof(v_tr, y[tr], seed=seed)
        x_va[TE_COL + "_te"] = te_apply(v_va, mp, prior)
    m = make_hgb_model(seed=seed + i, **overrides)
    m.fit(x_tr, y[tr])
    p = m.predict_proba(x_va)[:, 1]
    return i, va, p, auc(y[va], p), m


def run_cv_hgb(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    model_overrides: dict | None = None,
    freq: bool = True,
    te: bool = True,
    test: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> CvResult:
    """Stratified HistGradientBoosting CV on Deotte features (freq + TE)."""
    y = encode_target(df[TARGET]).to_numpy()
    fb = FeatureBuilder(freq=freq, te=te).fit(df, test)
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(skf.split(df, y))
    n_workers = resolve_max_workers(len(splits), max_workers)
    overrides = dict(model_overrides or {})
    jobs = [(df, y, fb, tr, va, i, seed, overrides) for i, (tr, va) in enumerate(splits)]
    rows = sorted(map_jobs(_hgb_one_fold, jobs, n_workers), key=lambda r: r[0])
    models, scores = [], []
    for _i, va, p, score, m in rows:
        oof[va] = p
        scores.append(score)
        models.append(m)
    mean, std = mean_std(scores)
    params = {**HGB_DEFAULTS, **(model_overrides or {})}
    return CvResult(oof, scores, mean, std, models, None, params)


def train_hgb(
    df: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    model_overrides: dict | None = None,
    freq: bool = True,
    te: bool = True,
    test: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> CvResult:
    """Run HGB CV on Deotte features, persist artifacts, optionally log."""
    cv = run_cv_hgb(
        df,
        folds=folds,
        seed=seed,
        model_overrides=model_overrides,
        freq=freq,
        te=te,
        test=test,
        max_workers=max_workers,
    )
    save_run(df, cv, out=out)
    if log:
        log_experiment(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC (HGB): {fmt_cv(cv.mean, cv.std)}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv
