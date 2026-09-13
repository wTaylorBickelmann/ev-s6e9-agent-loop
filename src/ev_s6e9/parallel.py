"""Process-pool helpers for independent CV units without CPU oversubscription.

Caps per-worker ``n_jobs`` / ``thread_count`` so workers × threads ≈ visible CPUs.
A single worker keeps the model default (``n_jobs=-1``). Does not change seeds,
``tree_method``, or ``device``.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections.abc import Callable, Sequence
from typing import TypeVar

ENV_WORKERS = "EV_S6E9_TRAIN_WORKERS"
ENV_BACKEND = "EV_S6E9_TRAIN_BACKEND"

T = TypeVar("T")
R = TypeVar("R")


def available_cpus() -> int:
    """Logical CPUs visible to this process (affinity-aware on Linux)."""

    n = os.cpu_count() or 1
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    return max(1, n)


def resolve_max_workers(n_tasks: int, max_workers: int | None = None) -> int:
    """Workers = min(tasks, knob, CPUs). ``1`` means sequential with ``n_jobs=-1``."""

    if n_tasks <= 1:
        return 1
    if max_workers is None:
        raw = os.environ.get(ENV_WORKERS)
        max_workers = int(raw) if raw not in (None, "") else available_cpus()
    return max(1, min(int(max_workers), n_tasks, available_cpus()))


def per_worker_n_jobs(n_workers: int) -> int | None:
    """``None`` keeps ``n_jobs=-1``; else ``max(1, cpus // n_workers)``."""

    if n_workers <= 1:
        return None
    return max(1, available_cpus() // n_workers)


def apply_thread_cap(
    overrides: dict | None,
    n_jobs: int | None,
    *,
    keys: Sequence[str] = ("n_jobs",),
) -> dict:
    """Copy overrides; cap thread keys when a per-worker limit applies."""

    out = dict(overrides or {})
    if n_jobs is None:
        return out
    for key in keys:
        cur = out.get(key)
        if cur is None or cur == -1 or (isinstance(cur, int) and cur > n_jobs):
            out[key] = n_jobs
    return out


def map_cv_jobs(
    fn: Callable[[T], R],
    jobs: Sequence[T],
    *,
    max_workers: int | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    backend: str | None = None,
) -> list[R]:
    """Map ``fn`` over independent jobs; sequential when only one worker is used."""

    workers = resolve_max_workers(len(jobs), max_workers)
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(job) for job in jobs]

    kind = (backend or os.environ.get(ENV_BACKEND) or "process").strip().lower()
    if kind == "thread":
        pool: ThreadPoolExecutor | ProcessPoolExecutor = ThreadPoolExecutor(
            max_workers=workers, initializer=initializer, initargs=initargs
        )
    elif kind == "process":
        import multiprocessing as mp

        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=initializer,
            initargs=initargs,
        )
    else:
        raise ValueError(f"unknown train backend {kind!r}; use process or thread")

    with pool as ex:
        return list(ex.map(fn, jobs))
