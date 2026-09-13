"""Parallel Deotte CV matches sequential OOF/AUC; worker/n_jobs knobs stay safe."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ev_s6e9.data import synth
from ev_s6e9.deotte import run_cv, run_cv_multi_seed
from ev_s6e9.features import FeatureBuilder
from ev_s6e9.model import XGB_DEFAULTS
from ev_s6e9.parallel import (
    apply_thread_cap,
    available_cpus,
    map_cv_jobs,
    per_worker_n_jobs,
    resolve_max_workers,
)


def test_resolve_max_workers_defaults_and_env(monkeypatch):
    """Default workers = min(tasks, CPUs); env/arg can lower that; one task stays 1."""

    monkeypatch.setattr("ev_s6e9.parallel.available_cpus", lambda: 8)
    monkeypatch.delenv("EV_S6E9_TRAIN_WORKERS", raising=False)
    assert resolve_max_workers(1) == 1
    assert resolve_max_workers(3) == 3
    assert resolve_max_workers(30) == 8
    monkeypatch.setenv("EV_S6E9_TRAIN_WORKERS", "4")
    assert resolve_max_workers(30) == 4
    assert resolve_max_workers(2) == 2
    assert resolve_max_workers(30, max_workers=3) == 3


def test_per_worker_n_jobs_and_thread_cap(monkeypatch):
    """One worker keeps n_jobs=-1; N workers cap so N * n_jobs ≈ CPUs."""

    monkeypatch.setattr("ev_s6e9.parallel.available_cpus", lambda: 16)
    assert per_worker_n_jobs(1) is None
    assert per_worker_n_jobs(4) == 4
    assert per_worker_n_jobs(16) == 1
    raw = {"max_depth": 6, "n_jobs": -1}
    assert apply_thread_cap(raw, None) == raw
    assert apply_thread_cap(raw, 2)["n_jobs"] == 2
    assert raw["n_jobs"] == -1
    assert apply_thread_cap({"n_jobs": 1}, 4)["n_jobs"] == 1
    assert apply_thread_cap(None, 3, keys=("thread_count",))["thread_count"] == 3


def test_map_cv_jobs_sequential_and_thread():
    """Sequential and thread backends return results in job order."""

    jobs = [3, 1, 4]
    assert map_cv_jobs(lambda x: x * 2, jobs, max_workers=1) == [6, 2, 8]
    assert map_cv_jobs(lambda x: x * 2, jobs, max_workers=3, backend="thread") == [6, 2, 8]


def test_available_cpus_positive():
    assert available_cpus() >= 1


def _stub_fit_fold(fb, variant, raw_tr, raw_va, y_tr, y_va, *, seed, overrides):
    """Deterministic fake fold: proba from ids + seed + variant (no XGB)."""

    ids = raw_va["id"].to_numpy(dtype=float)
    code = {"m1": 1.0, "m2": 2.0, "m3": 3.0}[variant.value]
    raw = np.mod(ids * 17.0 + float(seed) * 13.0 + code * 7.0, 97.0) / 97.0
    p = np.where(y_va == 1, 0.5 + 0.5 * raw, 0.5 * raw)
    return SimpleNamespace(n_jobs=overrides.get("n_jobs")), p


def test_thread_pool_matches_sequential_oof_auc(monkeypatch):
    """Stubbed folds: parallel (thread) OOF/AUC match sequential within tolerance."""

    monkeypatch.setattr("ev_s6e9.deotte._fit_fold", _stub_fit_fold)
    monkeypatch.setattr("ev_s6e9.parallel.available_cpus", lambda: 8)
    monkeypatch.delenv("EV_S6E9_TRAIN_WORKERS", raising=False)
    tr = synth(80, seed=1, target=True)
    te = synth(20, seed=2, target=False, start_id=10_000)
    kw = dict(folds=2, fold_seed=42, seeds=[42, 43], freq=True, te=True)
    seq = run_cv_multi_seed(tr, te, max_workers=1, **kw)
    par = run_cv_multi_seed(tr, te, max_workers=4, backend="thread", **kw)
    np.testing.assert_allclose(seq.oof, par.oof, atol=1e-12)
    assert seq.mean == pytest.approx(par.mean, abs=1e-12)
    assert seq.std == pytest.approx(par.std, abs=1e-12)
    assert seq.fold_aucs == pytest.approx(par.fold_aucs, abs=1e-12)
    assert seq.params.get("n_jobs") == XGB_DEFAULTS["n_jobs"] == -1
    assert {m.n_jobs for vc in par.variants.values() for m in vc.models} == {2}


def test_feature_builder_fitted_once(monkeypatch):
    """Workers reuse the parent-fitted FeatureBuilder; fit is not called per job."""

    monkeypatch.setattr("ev_s6e9.deotte._fit_fold", _stub_fit_fold)
    fits: list[int] = []
    orig = FeatureBuilder.fit

    def counting_fit(self, *args, **kwargs):
        fits.append(1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(FeatureBuilder, "fit", counting_fit)
    tr = synth(40, seed=0, target=True)
    run_cv_multi_seed(
        tr,
        folds=2,
        fold_seed=0,
        seeds=[1, 2],
        max_workers=3,
        backend="thread",
    )
    assert fits == [1]


def test_run_cv_parallel_matches_sequential(monkeypatch):
    """Single-seed run_cv also parallelizes the three variants."""

    monkeypatch.setattr("ev_s6e9.deotte._fit_fold", _stub_fit_fold)
    tr = synth(60, seed=3, target=True)
    seq = run_cv(tr, folds=2, seed=7, max_workers=1)
    par = run_cv(tr, folds=2, seed=7, max_workers=3, backend="thread")
    np.testing.assert_allclose(seq.oof, par.oof, atol=1e-12)
    assert seq.mean == pytest.approx(par.mean, abs=1e-12)


def test_process_pool_matches_sequential_tiny_xgb():
    """Spawned processes + tiny real XGB match sequential when n_jobs is fixed."""

    tr = synth(64, seed=4, target=True)
    te = synth(16, seed=5, target=False, start_id=2000)
    ov = {"n_estimators": 8, "early_stopping_rounds": 3, "max_depth": 3, "n_jobs": 1}
    kw = dict(folds=2, fold_seed=42, seeds=[42, 43], model_overrides=ov, freq=False, te=False)
    seq = run_cv_multi_seed(tr, te, max_workers=1, **kw)
    par = run_cv_multi_seed(tr, te, max_workers=2, backend="process", **kw)
    np.testing.assert_allclose(seq.oof, par.oof, atol=1e-10)
    assert seq.mean == pytest.approx(par.mean, abs=1e-10)
    assert list(seq.variants) == list(par.variants) == ["m1", "m2", "m3"]


def test_cli_exposes_max_workers():
    from ev_s6e9.__main__ import _parser

    args = _parser().parse_args(["train", "--strategy", "deotte", "--max-workers", "4"])
    assert args.max_workers == 4
