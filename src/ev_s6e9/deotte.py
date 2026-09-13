"""Deotte Fable 5.1: 3-model XGB blend (baseline, base_margin, recipe_feature).

Independent (seed, variant) CVs run in a process pool. Caller ``model_overrides``
pass through unchanged so XGB keeps ``n_jobs=-1`` (oversubscription is OK).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ev_s6e9.experiments import append_chunk, format_chunk
from ev_s6e9.features import FeatureBuilder, TE_COL, encode_target, te_apply, te_fit, te_oof, te_oof_avg
from ev_s6e9.metrics import auc, fmt_cv, mean_std
from ev_s6e9.model import XGB_DEFAULTS, make_xgb_model, short_xgb_params
from ev_s6e9.parallel import map_cv_jobs
from ev_s6e9.paths import CV_JSON, OOF_CSV, OUTPUTS, ROOT
from ev_s6e9.schema import ID_COL, TARGET

STRATEGY = "deotte"
VARIANTS = ("m1", "m2", "m3")

# Default OOFs for the s027 ensemble: LightGBM (exp0026) + Deotte 3-seed XGB (exp0024).
ENSEMBLE_OOF_A = ROOT / "exps" / "exp0026" / "oof.csv"  # LightGBM standalone
ENSEMBLE_OOF_B = ROOT / "exps" / "exp0024" / "oof.csv"  # Deotte 3-seed XGB blend

# Default OOFs for the s039 3-model rank-blend: XGB (exp0038) + LGBM (exp0026) + CatBoost (exp0029).
RANK_BLEND_OOF_A = ROOT / "exps" / "exp0038" / "oof.csv"  # s038 10-seed Deotte XGB
RANK_BLEND_OOF_B = ROOT / "exps" / "exp0026" / "oof.csv"  # s026 LightGBM standalone
RANK_BLEND_OOF_C = ROOT / "exps" / "exp0029" / "oof.csv"  # s029 CatBoost standalone


class DeotteVariant(Enum):
    """m1 baseline / m2 recipe as base_margin / m3 recipe as a feature."""

    BASELINE = "m1"
    BASE_MARGIN = "m2"
    RECIPE_FEATURE = "m3"


@dataclass
class VariantCv:
    """OOF, fold AUCs, and fold models for one of the three XGB variants."""

    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    models: list = field(default_factory=list)


@dataclass
class DeotteCvResult:
    """Equal-weight blend of m1/m2/m3 plus the fitted FeatureBuilder."""

    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    variants: dict[str, VariantCv]
    feature_builder: FeatureBuilder
    params: dict = field(default_factory=dict)


@dataclass
class EnsembleCvResult:
    """Grid-searched alpha blend of two pre-computed OOFs (LightGBM + Deotte).

    `alpha` is the weight on `oof_a`; the blend is `alpha * oof_a + (1 - alpha) * oof_b`.
    """

    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    alpha: float
    oof_a_path: str
    oof_b_path: str


@dataclass
class RankBlendCvResult:
    """Grid-searched 3-way rank-space blend of pre-computed OOFs.

    `weights` are non-negative simplex weights (sum=1) found by maximising mean
    fold AUC of the rank-space blend. `prob_*` fields are the control comparison:
    the same weights applied to raw probabilities.
    """

    oof: np.ndarray
    fold_aucs: list[float]
    mean: float
    std: float
    weights: list[float]
    oof_paths: list[str]
    prob_fold_aucs: list[float]
    prob_mean: float
    prob_std: float


def _fit_fold(
    fb: FeatureBuilder,
    variant: DeotteVariant,
    raw_tr: pd.DataFrame,
    raw_va: pd.DataFrame,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    *,
    seed: int,
    overrides: dict,
) -> tuple[object, np.ndarray]:
    """Fit one fold of one variant; return (model, val probabilities)."""

    use_margin = variant == DeotteVariant.BASE_MARGIN
    with_recipe = variant == DeotteVariant.RECIPE_FEATURE
    x_tr = fb.transform(raw_tr, with_recipe=with_recipe)
    x_va = fb.transform(raw_va, with_recipe=with_recipe)
    m = make_xgb_model(seed=seed, **overrides)
    if fb.te:
        v_tr = pd.to_numeric(raw_tr[TE_COL], errors="coerce")
        v_va = pd.to_numeric(raw_va[TE_COL], errors="coerce")
        mp, prior = te_fit(v_tr, y_tr)
        x_tr[TE_COL + "_te"] = te_oof_avg(v_tr, y_tr, m=20.0)
        x_va[TE_COL + "_te"] = te_apply(v_va, mp, prior)
        m.te_map_ = (mp, prior)
        # Second TE with lower regularisation (m=5) for a less-smoothed view
        mp5, prior5 = te_fit(v_tr, y_tr, m=5.0)
        x_tr[TE_COL + "_te_m5"] = te_oof_avg(v_tr, y_tr, m=5.0)
        x_va[TE_COL + "_te_m5"] = te_apply(v_va, mp5, prior5)
        m.te_map_m5_ = (mp5, prior5)
        # Third TE with near-zero smoothing (m=2) for a fine-grained, value-specific view
        mp2, prior2 = te_fit(v_tr, y_tr, m=2.0)
        x_tr[TE_COL + "_te_m2"] = te_oof_avg(v_tr, y_tr, m=2.0)
        x_va[TE_COL + "_te_m2"] = te_apply(v_va, mp2, prior2)
        m.te_map_m2_ = (mp2, prior2)
        # Commute TE: triple-smoothing (m=20, m=5, m=2) on Daily_Commute_km
        c_tr = pd.to_numeric(raw_tr["Daily_Commute_km"], errors="coerce")
        c_va = pd.to_numeric(raw_va["Daily_Commute_km"], errors="coerce")
        cm, cp = te_fit(c_tr, y_tr)
        x_tr["Daily_Commute_km_te"] = te_oof_avg(c_tr, y_tr, m=20.0)
        x_va["Daily_Commute_km_te"] = te_apply(c_va, cm, cp)
        m.commute_te_map_ = (cm, cp)
        cm5, cp5 = te_fit(c_tr, y_tr, m=5.0)
        x_tr["Daily_Commute_km_te_m5"] = te_oof_avg(c_tr, y_tr, m=5.0)
        x_va["Daily_Commute_km_te_m5"] = te_apply(c_va, cm5, cp5)
        m.commute_te_map_m5_ = (cm5, cp5)
        cm2, cp2 = te_fit(c_tr, y_tr, m=2.0)
        x_tr["Daily_Commute_km_te_m2"] = te_oof_avg(c_tr, y_tr, m=2.0)
        x_va["Daily_Commute_km_te_m2"] = te_apply(c_va, cm2, cp2)
        m.commute_te_map_m2_ = (cm2, cp2)
    if use_margin:
        margin_tr = fb.recipe_logit(raw_tr)
        margin_va = fb.recipe_logit(raw_va)
        m.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            base_margin=margin_tr,
            base_margin_eval_set=[margin_va],
            verbose=False,
        )
        p = m.predict_proba(x_va, base_margin=margin_va)[:, 1]
    else:
        m.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        p = m.predict_proba(x_va)[:, 1]
    return m, p


def run_variant_cv(
    df: pd.DataFrame,
    fb: FeatureBuilder,
    variant: DeotteVariant,
    *,
    folds: int = 5,
    seed: int = 42,
    model_seed: int | None = None,
    model_overrides: dict | None = None,
) -> VariantCv:
    """Stratified CV for a single Deotte variant.

    `seed` controls the fold split; `model_seed` (defaults to `seed`) controls
    XGB random_state so multi-seed blends can reuse identical folds.
    """

    if model_seed is None:
        model_seed = seed
    y = encode_target(df[TARGET]).to_numpy()
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models, scores = [], []
    overrides = dict(model_overrides or {})
    for i, (tr, va) in enumerate(skf.split(df, y)):
        m, p = _fit_fold(
            fb,
            variant,
            df.iloc[tr],
            df.iloc[va],
            y[tr],
            y[va],
            seed=model_seed + i,
            overrides=overrides,
        )
        oof[va] = p
        scores.append(auc(y[va], p))
        models.append(m)
    mean, std = mean_std(scores)
    return VariantCv(oof, scores, mean, std, models)


# Fitted FeatureBuilder + train frame, set once per worker (do not refit).
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class _VariantJob:
    """One independent (model_seed, variant) CV unit for the process pool."""

    model_seed: int
    variant: str
    folds: int
    fold_seed: int
    overrides: dict


def _init_variant_worker(train: pd.DataFrame, fb: FeatureBuilder) -> None:
    """Share the already-fitted FeatureBuilder and train frame with this worker."""

    _WORKER["train"] = train
    _WORKER["fb"] = fb


def _run_variant_job(job: _VariantJob) -> tuple[int, str, VariantCv]:
    """Run one (seed, variant) CV in a worker using the shared fitted builder."""

    vc = run_variant_cv(
        _WORKER["train"],
        _WORKER["fb"],
        DeotteVariant(job.variant),
        folds=job.folds,
        seed=job.fold_seed,
        model_seed=job.model_seed,
        model_overrides=job.overrides,
    )
    return job.model_seed, job.variant, vc


def _run_seed_variants(
    train: pd.DataFrame,
    fb: FeatureBuilder,
    *,
    seeds: list[int],
    folds: int,
    fold_seed: int,
    model_overrides: dict | None,
    max_workers: int | None,
    backend: str | None = None,
) -> dict[int, dict[str, VariantCv]]:
    """Fit every (seed, variant) pair; sequential when only one worker is used."""

    overrides = dict(model_overrides or {})
    jobs = [
        _VariantJob(model_seed=s, variant=v.value, folds=folds, fold_seed=fold_seed, overrides=overrides)
        for s in seeds
        for v in DeotteVariant
    ]
    mapped = map_cv_jobs(
        _run_variant_job,
        jobs,
        max_workers=max_workers,
        initializer=_init_variant_worker,
        initargs=(train, fb),
        backend=backend,
    )
    out: dict[int, dict[str, VariantCv]] = {s: {} for s in seeds}
    for model_seed, variant, vc in mapped:
        out[model_seed][variant] = vc
    return out


def run_cv(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    model_overrides: dict | None = None,
    freq: bool = False,
    te: bool = False,
    max_workers: int | None = None,
    backend: str | None = None,
) -> DeotteCvResult:
    """Fit FeatureBuilder once, run m1/m2/m3 (possibly in parallel), blend equally."""
    fb = FeatureBuilder(freq=freq, te=te).fit(train, test)
    by_seed = _run_seed_variants(
        train,
        fb,
        seeds=[seed],
        folds=folds,
        fold_seed=seed,
        model_overrides=model_overrides,
        max_workers=max_workers,
        backend=backend,
    )
    variants = {v.value: by_seed[seed][v.value] for v in DeotteVariant}
    oofs = [variants[v.value].oof for v in DeotteVariant]
    blend_oof = np.mean(oofs, axis=0)
    y = encode_target(train[TARGET]).to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    blend_scores = []
    for tr, va in skf.split(train, y):
        blend_scores.append(auc(y[va], blend_oof[va]))
    mean, std = mean_std(blend_scores)
    params = {**XGB_DEFAULTS, **(model_overrides or {})}
    return DeotteCvResult(blend_oof, blend_scores, mean, std, variants, fb, params)


def search_blend_weights(
    variant_oofs: list[np.ndarray],
    y: np.ndarray,
    skf: StratifiedKFold,
    *,
    n_grid: int = 30,
) -> tuple[np.ndarray, float, list[float]]:
    """Grid-search non-negative weights (sum=1) maximizing mean fold AUC.

    For 3 variants, parameterises the simplex as (w1, w2) with w3 = 1 - w1 - w2.
    Returns (best_weights, best_mean_auc, best_fold_aucs).
    """
    n = len(variant_oofs)
    best_w = np.ones(n) / n
    best_auc = -1.0
    best_folds: list[float] = []

    if n == 3:
        steps = np.linspace(0, 1, n_grid)
        for w1 in steps:
            for w2 in steps:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(w3, 0.0)
                blend = w1 * variant_oofs[0] + w2 * variant_oofs[1] + w3 * variant_oofs[2]
                fold_aucs = [auc(y[va], blend[va]) for _, va in skf.split(np.zeros(len(y)), y)]
                mean_auc = float(np.mean(fold_aucs))
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_w = np.array([w1, w2, w3])
                    best_folds = fold_aucs
    else:
        blend = np.mean(variant_oofs, axis=0)
        best_folds = [auc(y[va], blend[va]) for _, va in skf.split(np.zeros(len(y)), y)]
        best_auc = float(np.mean(best_folds))

    return best_w, best_auc, best_folds


def run_cv_multi_seed(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    folds: int = 5,
    fold_seed: int = 42,
    seeds: list[int] | None = None,
    model_overrides: dict | None = None,
    freq: bool = False,
    te: bool = False,
    weight_search: bool = False,
    max_workers: int | None = None,
    backend: str | None = None,
) -> DeotteCvResult:
    """Run Deotte 3-variant CV for each seed; average OOFs; evaluate on fixed folds.

    When `weight_search` is True, grid-searches non-equal blend weights (w1, w2, w3)
    on the per-variant OOFs (averaged across seeds) to maximise CV AUC.
    Independent (seed, variant) units may run in parallel; FeatureBuilder is fitted
    once and reused. Logged params keep the caller's overrides (typically n_jobs=-1).
    """
    if seeds is None:
        seeds = [fold_seed]
    fb = FeatureBuilder(freq=freq, te=te).fit(train, test)
    y = encode_target(train[TARGET]).to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=fold_seed)

    by_seed = _run_seed_variants(
        train,
        fb,
        seeds=seeds,
        folds=folds,
        fold_seed=fold_seed,
        model_overrides=model_overrides,
        max_workers=max_workers,
        backend=backend,
    )
    n_seeds = len(seeds)
    variant_oof_sums: dict[str, np.ndarray] = {v.value: np.zeros(len(y), dtype=float) for v in DeotteVariant}
    for s in seeds:
        for v in DeotteVariant:
            variant_oof_sums[v.value] += by_seed[s][v.value].oof
    variant_oofs_avg = [variant_oof_sums[v.value] / n_seeds for v in DeotteVariant]

    if weight_search:
        weights, best_auc, best_folds = search_blend_weights(variant_oofs_avg, y, skf)
        blend_oof = weights[0] * variant_oofs_avg[0] + weights[1] * variant_oofs_avg[1] + weights[2] * variant_oofs_avg[2]
        blend_scores = best_folds
        mean, std = mean_std(blend_scores)
        params = {**XGB_DEFAULTS, **(model_overrides or {}), "seeds": seeds, "blend_weights": weights.tolist()}
    else:
        blend_oof = np.mean(variant_oofs_avg, axis=0)
        blend_scores = [auc(y[va], blend_oof[va]) for _, va in skf.split(train, y)]
        mean, std = mean_std(blend_scores)
        params = {**XGB_DEFAULTS, **(model_overrides or {}), "seeds": seeds}

    first_variants = {v.value: by_seed[seeds[0]][v.value] for v in DeotteVariant}
    return DeotteCvResult(blend_oof, blend_scores, mean, std, first_variants, fb, params)


def save_run(df: pd.DataFrame, cv: DeotteCvResult, out: Path | None = None) -> None:
    """Write OOF, per-variant fold models, FeatureBuilder, and cv.json."""

    out = out or OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    models_root = out / "models"
    models_root.mkdir(exist_ok=True)
    for name in VARIANTS:
        (models_root / name).mkdir(exist_ok=True)
    joblib.dump(cv.feature_builder, out / "feature_builder.joblib")
    pd.DataFrame({ID_COL: df[ID_COL], TARGET: cv.oof}).to_csv(out / OOF_CSV.name, index=False)
    for name, vc in cv.variants.items():
        for i, m in enumerate(vc.models):
            joblib.dump(m, models_root / name / f"fold{i}.joblib")
    payload = {
        "strategy": STRATEGY,
        "fold_aucs": cv.fold_aucs,
        "mean": cv.mean,
        "std": cv.std,
        "cv": fmt_cv(cv.mean, cv.std),
        "params": {k: v for k, v in cv.params.items() if k != "random_state"},
        "n": int(len(df)),
        "folds": len(cv.fold_aucs),
        "variants": {
            name: {"fold_aucs": vc.fold_aucs, "mean": vc.mean, "std": vc.std, "cv": fmt_cv(vc.mean, vc.std)}
            for name, vc in cv.variants.items()
        },
    }
    (out / CV_JSON.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_experiment(
    cv: DeotteCvResult,
    *,
    folds: int,
    note: str = "",
    path: Path | None = None,
) -> Path:
    """Append a Deotte blend chunk to EXPERIMENTS.md."""

    title = f"Deotte XGB 3-model blend ({short_xgb_params(cv.params)}), {folds}-fold"
    chunk = format_chunk(
        title,
        fmt_cv(cv.mean, cv.std),
        takeaway=note or "auto-logged from train --strategy deotte",
    )
    return append_chunk(chunk, path=path)


def train(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    seeds: list[int] | None = None,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    model_overrides: dict | None = None,
    freq: bool = False,
    te: bool = False,
    weight_search: bool = False,
    max_workers: int | None = None,
    backend: str | None = None,
) -> DeotteCvResult:
    """Run Deotte CV, persist artifacts, optionally append EXPERIMENTS.md."""
    if seeds:
        cv = run_cv_multi_seed(
            df, test, folds=folds, fold_seed=seed, seeds=seeds,
            model_overrides=model_overrides, freq=freq, te=te,
            weight_search=weight_search,
            max_workers=max_workers,
            backend=backend,
        )
    else:
        cv = run_cv(
            df, test, folds=folds, seed=seed, model_overrides=model_overrides,
            freq=freq, te=te, max_workers=max_workers, backend=backend,
        )
    save_run(df, cv, out=out)
    if log:
        log_experiment(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC (blend): {fmt_cv(cv.mean, cv.std)}")
    for name in VARIANTS:
        vc = cv.variants[name]
        print(f"  {name}: {fmt_cv(vc.mean, vc.std)}")
    if "blend_weights" in cv.params:
        w = cv.params["blend_weights"]
        print(f"  blend weights: m1={w[0]:.4f} m2={w[1]:.4f} m3={w[2]:.4f}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv


def _predict_variant(
    df: pd.DataFrame,
    fb: FeatureBuilder,
    variant: str,
    models: list,
) -> np.ndarray:
    """Average fold probabilities for one variant (margin-aware for m2)."""

    v = DeotteVariant(variant)
    use_margin = v == DeotteVariant.BASE_MARGIN
    with_recipe = v == DeotteVariant.RECIPE_FEATURE
    if use_margin:
        margin = fb.recipe_logit(df)
        ps = []
        for m in models:
            x = fb.transform(df, with_recipe=with_recipe)
            if fb.te:
                x[TE_COL + "_te"] = te_apply(
                    pd.to_numeric(df[TE_COL], errors="coerce"), *m.te_map_
                )
                x[TE_COL + "_te_m5"] = te_apply(
                    pd.to_numeric(df[TE_COL], errors="coerce"), *m.te_map_m5_
                )
                x[TE_COL + "_te_m2"] = te_apply(
                    pd.to_numeric(df[TE_COL], errors="coerce"), *m.te_map_m2_
                )
                x["Daily_Commute_km_te"] = te_apply(
                    pd.to_numeric(df["Daily_Commute_km"], errors="coerce"), *m.commute_te_map_
                )
                x["Daily_Commute_km_te_m5"] = te_apply(
                    pd.to_numeric(df["Daily_Commute_km"], errors="coerce"), *m.commute_te_map_m5_
                )
                x["Daily_Commute_km_te_m2"] = te_apply(
                    pd.to_numeric(df["Daily_Commute_km"], errors="coerce"), *m.commute_te_map_m2_
                )
            ps.append(m.predict_proba(x, base_margin=margin)[:, 1])
    else:
        ps = []
        for m in models:
            x = fb.transform(df, with_recipe=with_recipe)
            if fb.te:
                x[TE_COL + "_te"] = te_apply(
                    pd.to_numeric(df[TE_COL], errors="coerce"), *m.te_map_
                )
                x[TE_COL + "_te_m5"] = te_apply(
                    pd.to_numeric(df[TE_COL], errors="coerce"), *m.te_map_m5_
                )
                x[TE_COL + "_te_m2"] = te_apply(
                    pd.to_numeric(df[TE_COL], errors="coerce"), *m.te_map_m2_
                )
                x["Daily_Commute_km_te"] = te_apply(
                    pd.to_numeric(df["Daily_Commute_km"], errors="coerce"), *m.commute_te_map_
                )
                x["Daily_Commute_km_te_m5"] = te_apply(
                    pd.to_numeric(df["Daily_Commute_km"], errors="coerce"), *m.commute_te_map_m5_
                )
                x["Daily_Commute_km_te_m2"] = te_apply(
                    pd.to_numeric(df["Daily_Commute_km"], errors="coerce"), *m.commute_te_map_m2_
                )
            ps.append(m.predict_proba(x)[:, 1])
    return np.mean(ps, axis=0)


def predict_proba(df: pd.DataFrame, out: Path | None = None) -> np.ndarray:
    """Load saved Deotte artifacts and average m1/m2/m3 test probabilities."""

    out = out or OUTPUTS
    fb = joblib.load(out / "feature_builder.joblib")
    preds = []
    for name in VARIANTS:
        paths = sorted((out / "models" / name).glob("fold*.joblib"))
        if not paths:
            raise FileNotFoundError(f"no fold models in {out / 'models' / name}; run train --strategy deotte")
        models = [joblib.load(p) for p in paths]
        preds.append(_predict_variant(df, fb, name, models))
    return np.mean(preds, axis=0)


def _resolve_oof(path: Path | str | None, default: Path) -> Path:
    """Resolve an OOF path (absolute, or relative to repo root) with a default."""

    if path is None:
        return default
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def search_ensemble_weight(
    oof_a: np.ndarray,
    oof_b: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
    *,
    n_grid: int = 101,
) -> tuple[float, float, list[float]]:
    """Grid-search alpha in [0, 1] maximising mean fold AUC of the blend.

    Blend is `alpha * oof_a + (1 - alpha) * oof_b`. Returns
    (best_alpha, best_mean_auc, best_fold_aucs).
    """
    fold_idx = list(skf.split(np.zeros(len(y)), y))
    steps = np.linspace(0.0, 1.0, n_grid)
    best_alpha, best_auc, best_folds = 0.5, -1.0, []
    for alpha in steps:
        blend = alpha * oof_a + (1.0 - alpha) * oof_b
        fold_aucs = [auc(y[va], blend[va]) for _, va in fold_idx]
        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_alpha = float(alpha)
            best_folds = fold_aucs
    return best_alpha, best_auc, best_folds


def run_ensemble_cv(
    df: pd.DataFrame,
    oof_a: Path | str | None = None,
    oof_b: Path | str | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    n_grid: int = 101,
) -> EnsembleCvResult:
    """Blend two pre-computed OOFs on fixed folds; grid-search the weight alpha.

    OOFs are aligned to `df` by `id`. Evaluation uses the same
    `StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)` as the
    original CV so the ensemble AUC is comparable to the standalone runs.
    """
    a_path = _resolve_oof(oof_a, ENSEMBLE_OOF_A)
    b_path = _resolve_oof(oof_b, ENSEMBLE_OOF_B)
    y = encode_target(df[TARGET]).to_numpy()
    ids = df[ID_COL].to_numpy()

    def _load(path: Path) -> np.ndarray:
        d = pd.read_csv(path).set_index(ID_COL).reindex(ids)
        if d[TARGET].isna().any():
            raise ValueError(f"OOF {path} is missing ids present in train")
        return d[TARGET].to_numpy(dtype=float)

    oa = _load(a_path)
    ob = _load(b_path)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    alpha, _, best_folds = search_ensemble_weight(oa, ob, y, skf, n_grid=n_grid)
    blend_oof = alpha * oa + (1.0 - alpha) * ob
    mean, std = mean_std(best_folds)
    return EnsembleCvResult(
        oof=blend_oof,
        fold_aucs=best_folds,
        mean=mean,
        std=std,
        alpha=alpha,
        oof_a_path=str(a_path),
        oof_b_path=str(b_path),
    )


def save_ensemble_run(df: pd.DataFrame, cv: EnsembleCvResult, out: Path | None = None) -> None:
    """Write the blended OOF and a cv.json payload (no fold models)."""

    out = out or OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: df[ID_COL], TARGET: cv.oof}).to_csv(out / OOF_CSV.name, index=False)
    payload = {
        "strategy": "ensemble",
        "fold_aucs": cv.fold_aucs,
        "mean": cv.mean,
        "std": cv.std,
        "cv": fmt_cv(cv.mean, cv.std),
        "params": {
            "alpha": cv.alpha,
            "oof_a": cv.oof_a_path,
            "oof_b": cv.oof_b_path,
            "folds": len(cv.fold_aucs),
        },
        "n": int(len(df)),
        "folds": len(cv.fold_aucs),
    }
    (out / CV_JSON.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_ensemble(
    cv: EnsembleCvResult,
    *,
    folds: int,
    note: str = "",
    path: Path | None = None,
) -> Path:
    """Append an ensemble chunk to EXPERIMENTS.md."""

    title = f"Ensemble LGBM+Deotte (alpha={cv.alpha:.3f}), {folds}-fold"
    chunk = format_chunk(
        title,
        fmt_cv(cv.mean, cv.std),
        takeaway=note or "auto-logged from train --strategy ensemble",
    )
    return append_chunk(chunk, path=path)


def train_ensemble(
    df: pd.DataFrame,
    oof_a: Path | str | None = None,
    oof_b: Path | str | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    n_grid: int = 101,
) -> EnsembleCvResult:
    """Blend two pre-computed OOFs, persist artifacts, optionally log EXPERIMENTS.md."""

    cv = run_ensemble_cv(df, oof_a, oof_b, folds=folds, seed=seed, n_grid=n_grid)
    save_ensemble_run(df, cv, out=out)
    if log:
        log_ensemble(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC (ensemble): {fmt_cv(cv.mean, cv.std)}")
    print(f"  alpha (weight on oof_a={cv.oof_a_path}): {cv.alpha:.4f}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv


def _to_percentile_rank(oof: np.ndarray) -> np.ndarray:
    """Percentile ranks in (0, 1]: `scipy.stats.rankdata(oof) / len(oof)`."""

    from scipy.stats import rankdata

    return rankdata(oof) / len(oof)


def search_rank_blend_weights(
    rank_oofs: list[np.ndarray],
    y: np.ndarray,
    skf: StratifiedKFold,
    *,
    n_grid: int = 30,
) -> tuple[np.ndarray, float, list[float]]:
    """Grid-search non-negative 3-way simplex weights (sum=1) maximising mean fold AUC
    of the rank-space blend.

    Parameterises the simplex as (w1, w2) with w3 = 1 - w1 - w2. Returns
    (best_weights, best_mean_auc, best_fold_aucs).
    """
    n = len(rank_oofs)
    best_w = np.ones(n) / n
    best_auc = -1.0
    best_folds: list[float] = []
    fold_idx = list(skf.split(np.zeros(len(y)), y))
    steps = np.linspace(0.0, 1.0, n_grid)
    for w1 in steps:
        for w2 in steps:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = max(w3, 0.0)
            blend = w1 * rank_oofs[0] + w2 * rank_oofs[1] + w3 * rank_oofs[2]
            fold_aucs = [auc(y[va], blend[va]) for _, va in fold_idx]
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_w = np.array([w1, w2, w3])
                best_folds = fold_aucs
    return best_w, best_auc, best_folds


def run_rank_blend_cv(
    df: pd.DataFrame,
    oof_a: Path | str | None = None,
    oof_b: Path | str | None = None,
    oof_c: Path | str | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    n_grid: int = 30,
) -> RankBlendCvResult:
    """3-way rank-space blend of pre-computed OOFs; grid-search simplex weights.

    OOFs are aligned to `df` by `id`. Each OOF is converted to percentile ranks
    (`rankdata/len`); weights are searched on the rank-space blend to maximise mean
    fold AUC on `StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)`.
    The same weights are also applied to raw probabilities as a control comparison.
    """
    a_path = _resolve_oof(oof_a, RANK_BLEND_OOF_A)
    b_path = _resolve_oof(oof_b, RANK_BLEND_OOF_B)
    c_path = _resolve_oof(oof_c, RANK_BLEND_OOF_C)
    y = encode_target(df[TARGET]).to_numpy()
    ids = df[ID_COL].to_numpy()

    def _load(path: Path) -> np.ndarray:
        d = pd.read_csv(path).set_index(ID_COL).reindex(ids)
        if d[TARGET].isna().any():
            raise ValueError(f"OOF {path} is missing ids present in train")
        return d[TARGET].to_numpy(dtype=float)

    oofs = [_load(p) for p in (a_path, b_path, c_path)]
    rank_oofs = [_to_percentile_rank(o) for o in oofs]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    weights, _, best_folds = search_rank_blend_weights(rank_oofs, y, skf, n_grid=n_grid)
    blend_oof = weights[0] * rank_oofs[0] + weights[1] * rank_oofs[1] + weights[2] * rank_oofs[2]
    mean, std = mean_std(best_folds)
    # Control: the same simplex weights applied to raw probabilities.
    prob_blend = weights[0] * oofs[0] + weights[1] * oofs[1] + weights[2] * oofs[2]
    prob_folds = [auc(y[va], prob_blend[va]) for _, va in skf.split(np.zeros(len(y)), y)]
    prob_mean, prob_std = mean_std(prob_folds)
    return RankBlendCvResult(
        oof=blend_oof,
        fold_aucs=best_folds,
        mean=mean,
        std=std,
        weights=weights.tolist(),
        oof_paths=[str(a_path), str(b_path), str(c_path)],
        prob_fold_aucs=prob_folds,
        prob_mean=prob_mean,
        prob_std=prob_std,
    )


def save_rank_blend_run(df: pd.DataFrame, cv: RankBlendCvResult, out: Path | None = None) -> None:
    """Write the rank-blended OOF and a cv.json payload (no fold models)."""

    out = out or OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: df[ID_COL], TARGET: cv.oof}).to_csv(out / OOF_CSV.name, index=False)
    payload = {
        "strategy": "rank_blend",
        "fold_aucs": cv.fold_aucs,
        "mean": cv.mean,
        "std": cv.std,
        "cv": fmt_cv(cv.mean, cv.std),
        "params": {
            "weights": cv.weights,
            "oof_a": cv.oof_paths[0],
            "oof_b": cv.oof_paths[1],
            "oof_c": cv.oof_paths[2],
            "folds": len(cv.fold_aucs),
            "prob_blend_cv": fmt_cv(cv.prob_mean, cv.prob_std),
            "prob_blend_mean": cv.prob_mean,
        },
        "n": int(len(df)),
        "folds": len(cv.fold_aucs),
    }
    (out / CV_JSON.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_rank_blend(
    cv: RankBlendCvResult,
    *,
    folds: int,
    note: str = "",
    path: Path | None = None,
) -> Path:
    """Append a rank-blend chunk to EXPERIMENTS.md."""

    w = cv.weights
    title = f"3-model OOF rank-blend (w={w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}), {folds}-fold"
    chunk = format_chunk(
        title,
        fmt_cv(cv.mean, cv.std),
        takeaway=note or f"rank-blend {fmt_cv(cv.mean, cv.std)}; prob-blend control {fmt_cv(cv.prob_mean, cv.prob_std)}",
    )
    return append_chunk(chunk, path=path)


def train_rank_blend(
    df: pd.DataFrame,
    oof_a: Path | str | None = None,
    oof_b: Path | str | None = None,
    oof_c: Path | str | None = None,
    *,
    folds: int = 5,
    seed: int = 42,
    log: bool = True,
    note: str = "",
    experiments_path: Path | None = None,
    out: Path | None = None,
    n_grid: int = 30,
) -> RankBlendCvResult:
    """3-way rank-space OOF blend, persist artifacts, optionally log EXPERIMENTS.md."""

    cv = run_rank_blend_cv(df, oof_a, oof_b, oof_c, folds=folds, seed=seed, n_grid=n_grid)
    save_rank_blend_run(df, cv, out=out)
    if log:
        log_rank_blend(cv, folds=folds, note=note, path=experiments_path)
    print(f"CV AUC (rank-blend): {fmt_cv(cv.mean, cv.std)}")
    print(f"  weights: a={cv.weights[0]:.4f} b={cv.weights[1]:.4f} c={cv.weights[2]:.4f}")
    print(f"  prob-blend control: {fmt_cv(cv.prob_mean, cv.prob_std)}")
    print("folds:", ", ".join(f"{a:.5f}" for a in cv.fold_aucs))
    return cv
