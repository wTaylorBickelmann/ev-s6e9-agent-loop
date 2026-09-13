"""Process-parallel CV matches sequential OOF/AUC; worker knobs cap threads."""

from __future__ import annotations

import numpy as np
import pytest

from ev_s6e9 import deotte as deotte_mod
from ev_s6e9 import parallel as par
from ev_s6e9.__main__ import _parser
from ev_s6e9.data import synth
from ev_s6e9.deotte import run_cv, run_cv_multi_seed
from ev_s6e9.features import FeatureBuilder
from ev_s6e9.model import XGB_DEFAULTS
from ev_s6e9.train import run_cv as run_lgbm_cv


class _StubModel:
    """Stand-in fold model so tests skip real XGB."""

    def __init__(self, seed: int, variant: str) -> None:
        self.seed = seed
        self.variant = variant


def _stub_fit_fold(fb, variant, raw_tr, raw_va, y_tr, y_va, *, seed, overrides):
    """Deterministic val probs from row index, model seed, and variant (no XGB)."""

    vcode = {"m1": 1, "m2": 2, "m3": 3}[variant.value]
    idx = raw_va.index.to_numpy(dtype=np.int64)
    p = ((idx * 17 + seed * 31 + vcode * 13) % 997) / 997.0
    return _StubModel(seed, variant.value), p.astype(float)


def _tiny_train(n: int = 48, seed: int = 0):
    """Small synth frame with both classes for 2-fold stratified CV."""

    return synth(n, seed=seed, target=True)


def test_resolve_max_workers_clamps(monkeypatch):
    monkeypatch.setattr(par, "cpu_count", lambda: 16)
    monkeypatch.delenv(par.ENV_WORKERS, raising=False)
    assert par.resolve_max_workers(1, 8) == 1
    assert par.resolve_max_workers(9, 4) == 4
    assert par.resolve_max_workers(9, 100) == 9
    assert par.resolve_max_workers(9, 0) == 1
    assert par.resolve_max_workers(9, None) == 9
    monkeypatch.setattr(par, "cpu_count", lambda: 4)
    assert par.resolve_max_workers(9, None) == 4


def test_resolve_max_workers_env(monkeypatch):
    monkeypatch.setattr(par, "cpu_count", lambda: 16)
    monkeypatch.setenv(par.ENV_WORKERS, "2")
    assert par.resolve_max_workers(9, None) == 2
    assert par.resolve_max_workers(9, 3) == 3


def test_per_worker_n_jobs_single_and_cap(monkeypatch):
    monkeypatch.setattr(par, "cpu_count", lambda: 16)
    assert par.per_worker_n_jobs(1) == -1
    assert par.per_worker_n_jobs(1, 4) == 4
    assert par.per_worker_n_jobs(4) == 4
    assert par.per_worker_n_jobs(4, 2) == 2
    assert par.per_worker_n_jobs(4, 8) == 4


def test_cap_model_threads_skips_single_worker_and_leaves_xgb_keys():
    raw = {"tree_method": "hist", "device": "cpu", "max_depth": 6}
    assert par.cap_model_threads(raw, 1) == raw
    capped = par.cap_model_threads(raw, 4)
    assert capped["tree_method"] == "hist"
    assert capped["device"] == "cpu"
    assert capped["n_jobs"] == par.per_worker_n_jobs(4)
    assert raw.get("n_jobs") is None


def test_deotte_parallel_matches_sequential_stubbed(monkeypatch):
    monkeypatch.setenv(par.ENV_BACKEND, "threads")
    monkeypatch.setattr(deotte_mod, "_fit_fold", _stub_fit_fold)
    df = _tiny_train()
    seq = run_cv(df, folds=2, seed=0, max_workers=1)
    par_cv = run_cv(df, folds=2, seed=0, max_workers=3)
    np.testing.assert_allclose(seq.oof, par_cv.oof, atol=1e-12)
    assert seq.mean == pytest.approx(par_cv.mean)
    assert seq.fold_aucs == pytest.approx(par_cv.fold_aucs)
    for name in ("m1", "m2", "m3"):
        np.testing.assert_allclose(seq.variants[name].oof, par_cv.variants[name].oof, atol=1e-12)
        assert seq.variants[name].fold_aucs == pytest.approx(par_cv.variants[name].fold_aucs)


def test_deotte_multi_seed_parallel_matches_sequential(monkeypatch):
    monkeypatch.setenv(par.ENV_BACKEND, "threads")
    monkeypatch.setattr(deotte_mod, "_fit_fold", _stub_fit_fold)
    df = _tiny_train()
    kwargs = dict(folds=2, fold_seed=0, seeds=[7, 11], freq=True, te=True)
    seq = run_cv_multi_seed(df, max_workers=1, **kwargs)
    par_cv = run_cv_multi_seed(df, max_workers=4, **kwargs)
    np.testing.assert_allclose(seq.oof, par_cv.oof, atol=1e-12)
    assert seq.mean == pytest.approx(par_cv.mean)
    assert seq.fold_aucs == pytest.approx(par_cv.fold_aucs)


def test_feature_builder_fit_once_across_seeds(monkeypatch):
    monkeypatch.setenv(par.ENV_BACKEND, "threads")
    monkeypatch.setattr(deotte_mod, "_fit_fold", _stub_fit_fold)
    calls = {"n": 0}
    orig = FeatureBuilder.fit

    def _count(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(FeatureBuilder, "fit", _count)
    df = _tiny_train()
    cv = run_cv_multi_seed(df, folds=2, fold_seed=0, seeds=[1, 2, 3], max_workers=3)
    assert calls["n"] == 1
    assert cv.feature_builder is not None
    assert cv.feature_builder._fitted


def test_parallel_caps_n_jobs_single_worker_keeps_default(monkeypatch):
    monkeypatch.setenv(par.ENV_BACKEND, "threads")
    monkeypatch.setattr(par, "cpu_count", lambda: 12)
    seen: list[int | None] = []

    def _capture(fb, variant, raw_tr, raw_va, y_tr, y_va, *, seed, overrides):
        seen.append(overrides.get("n_jobs"))
        return _stub_fit_fold(fb, variant, raw_tr, raw_va, y_tr, y_va, seed=seed, overrides=overrides)

    monkeypatch.setattr(deotte_mod, "_fit_fold", _capture)
    df = _tiny_train()
    run_cv(df, folds=2, seed=0, max_workers=1)
    assert all(j is None for j in seen)
    seen.clear()
    run_cv(df, folds=2, seed=0, max_workers=3)
    assert seen
    assert all(j == par.per_worker_n_jobs(3) for j in seen)
    assert seen[0] != -1


def test_logged_params_keep_hist_cpu_and_default_n_jobs(monkeypatch):
    monkeypatch.setenv(par.ENV_BACKEND, "threads")
    monkeypatch.setattr(deotte_mod, "_fit_fold", _stub_fit_fold)
    df = _tiny_train()
    cv = run_cv(df, folds=2, seed=0, max_workers=3)
    assert cv.params["tree_method"] == "hist"
    assert cv.params["device"] == "cpu"
    assert cv.params["n_jobs"] == XGB_DEFAULTS["n_jobs"] == -1


def test_deotte_process_pool_matches_sequential():
    """Real spawn pool + tiny XGB; pin n_jobs=1 so thread count matches sequential."""

    df = _tiny_train(n=56, seed=2)
    ov = {"n_estimators": 8, "early_stopping_rounds": 3, "n_jobs": 1}
    seq = run_cv(df, folds=2, seed=0, model_overrides=ov, max_workers=1)
    par_cv = run_cv(df, folds=2, seed=0, model_overrides=ov, max_workers=3)
    np.testing.assert_allclose(seq.oof, par_cv.oof, atol=1e-8)
    assert seq.mean == pytest.approx(par_cv.mean, abs=1e-8)
    assert seq.params["tree_method"] == par_cv.params["tree_method"] == "hist"
    assert seq.params["device"] == par_cv.params["device"] == "cpu"


def test_lgbm_parallel_folds_match_sequential(monkeypatch):
    monkeypatch.setenv(par.ENV_BACKEND, "threads")
    df = _tiny_train()
    ov = {"n_estimators": 15, "n_jobs": 1}
    seq = run_lgbm_cv(df, folds=2, seed=0, model_overrides=ov, max_workers=1)
    par_cv = run_lgbm_cv(df, folds=2, seed=0, model_overrides=ov, max_workers=2)
    np.testing.assert_allclose(seq.oof, par_cv.oof, atol=1e-10)
    assert seq.mean == pytest.approx(par_cv.mean)
    assert seq.fold_aucs == pytest.approx(par_cv.fold_aucs)


def test_cli_accepts_max_workers():
    args = _parser().parse_args(["train", "--strategy", "deotte", "--max-workers", "4"])
    assert args.max_workers == 4
