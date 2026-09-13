"""Worker-count and thread-cap helpers for process-parallel CV.

Independent seeds, variants, and folds can run in a process pool. Each worker
gets a thread cap so `N * n_jobs` stays near the machine CPU count. A single
worker keeps the historical `n_jobs=-1` (all cores) default.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from typing import Any, Callable, Sequence

ENV_WORKERS = "EV_S6E9_TRAIN_WORKERS"
ENV_BACKEND = "EV_S6E9_TRAIN_BACKEND"


def cpu_count() -> int:
    """Usable CPUs (affinity when present), at least 1."""

    try:
        n = len(os.sched_getaffinity(0))
        return n or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_max_workers(n_tasks: int, max_workers: int | None = None) -> int:
    """Clamp workers to `[1, n_tasks]` from arg, `EV_S6E9_TRAIN_WORKERS`, or `min(tasks, CPUs)`."""

    if n_tasks <= 1:
        return 1
    requested = max_workers
    if requested is None:
        raw = os.environ.get(ENV_WORKERS)
        requested = int(raw) if raw not in (None, "") else min(n_tasks, cpu_count())
    return max(1, min(n_tasks, int(requested)))


def per_worker_n_jobs(n_workers: int, explicit: int | None = None) -> int:
    """One worker keeps `n_jobs=-1` (or a positive explicit). Else `N * n_jobs ≈ cpu_count`."""

    if n_workers <= 1:
        return explicit if (explicit is not None and explicit > 0) else -1
    cap = max(1, cpu_count() // n_workers)
    if explicit is not None and explicit > 0:
        return min(explicit, cap)
    return cap


def cap_model_threads(
    overrides: dict | None,
    n_workers: int,
    *,
    keys: tuple[str, ...] = ("n_jobs",),
) -> dict[str, Any]:
    """Copy overrides; multi-worker paths cap `n_jobs` / `thread_count`. Single-worker is unchanged."""

    ov = dict(overrides or {})
    if n_workers <= 1:
        return ov
    for key in keys:
        raw = ov.get(key)
        expl = raw if isinstance(raw, int) and raw > 0 else None
        ov[key] = per_worker_n_jobs(n_workers, expl)
    return ov


def map_jobs(
    fn: Callable,
    jobs: Sequence,
    n_workers: int,
    *,
    initializer: Callable | None = None,
    initargs: tuple = (),
) -> list:
    """Apply `fn` to each job in submission order. Sequential if one worker; else spawn (or threads)."""

    backend = os.environ.get(ENV_BACKEND, "processes")
    if n_workers <= 1 or backend in ("sequential", "seq"):
        if initializer is not None:
            initializer(*initargs)
        return [fn(job) for job in jobs]
    if backend in ("threads", "threading", "thread"):
        if initializer is not None:
            initializer(*initargs)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(fn, jobs))
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        return list(pool.map(fn, jobs))
