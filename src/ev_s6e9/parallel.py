"""Process-pool helpers for independent CV units.

Workers parallelize (seed, variant, fold) jobs. Training keeps model thread
defaults (``n_jobs=-1``); oversubscription is acceptable. Does not change seeds,
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
    """Workers = min(tasks, knob, CPUs). ``1`` means sequential (no pool)."""

    if n_tasks <= 1:
        return 1
    if max_workers is None:
        raw = os.environ.get(ENV_WORKERS)
        max_workers = int(raw) if raw not in (None, "") else available_cpus()
    return max(1, min(int(max_workers), n_tasks, available_cpus()))


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
