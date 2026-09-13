"""Git helpers for the loop: commit every run and rewind to the best baseline.

Two behaviours wired into ``Loop.run`` (live mode only, skipped when the root is
not a git repo so dry-run tests on temp dirs stay green):

1. ``rewind_to_best`` — at the START of every iteration, reset the working tree of
   the code the executor touches (``exps/`` and ``src/``) to the latest best-keep
   commit (``loop: KEEP``). A failed run's residue therefore never contaminates the
   next run, which always starts from proven-good code. The ``ledger/`` and memory
   markdown (RESULTS, STRATEGIES, EXPERIMENTS, LEARNINGS, …) are deliberately NOT
   rewound: that is the loop's cumulative memory and must keep growing.

2. ``commit_run`` — after ``record``, git-commit every allow-listed changed path
   (same deny-list as submit-if-improved, so data CSVs / OOF / models / secrets are
   never staged) with a message carrying the strategy id, status, and CV. This runs
   for BOTH ok and fail runs, so nothing is lost even when an experiment breaks.

Both are no-ops (returning without side effects) when the root is not a git repo
or has no best-keep commit yet.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loop.submit_if_improved import stage_allowed

# Paths the executor is allowed to touch and that we rewind to the best baseline.
# `ledger/` and the root memory markdown are intentionally excluded: rewinding them
# would (a) drop committed fail rows from the working tree and (b) shrink the next
# strategy id back onto already-used ids, causing collisions.
_REWIND_PATHS = ("exps", "src")


def is_git_repo(root: Path, *, run=subprocess.run) -> bool:
    """True when ``root`` is inside a git work tree."""
    proc = run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0 and (proc.stdout or "").strip() == "true"


def best_baseline_commit(root: Path, *, run=subprocess.run) -> str | None:
    """Latest ``loop: KEEP`` commit SHA, or None if none exists yet."""
    proc = run(
        ["git", "log", "-1", "--format=%H", "--grep=loop: KEEP"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    sha = (proc.stdout or "").strip()
    return sha or None


def rewind_to_best(
    root: Path,
    *,
    paths: tuple[str, ...] = _REWIND_PATHS,
    run=subprocess.run,
) -> list[str] | None:
    """Reset ``paths`` to the latest best-keep commit. Returns restored paths or None.

    No-op when the root is not a git repo or there is no keep commit yet. Restores
    only ``exps/`` and ``src/`` so the executor starts from proven-good code while
    the ledger (loop memory) keeps accumulating.

    ``git checkout <baseline> -- <path>`` restores files that exist in the baseline
    but does NOT delete worktree files absent from it, so we additionally remove any
    allow-listed worktree file under ``paths`` that is not tracked in the baseline —
    otherwise a failed run's leftover exp dirs would survive into the next run.
    """
    if not is_git_repo(root, run=run):
        return None
    baseline = best_baseline_commit(root, run=run)
    if baseline is None:
        return None

    # Files tracked in the baseline under the rewind paths (relative, no ./).
    ls = run(
        ["git", "ls-tree", "-r", "--name-only", baseline, "--", *paths],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    baseline_files = {p for p in (ls.stdout or "").splitlines() if p}

    # Files currently tracked in the index under the rewind paths.
    idx = run(
        ["git", "ls-files", "--", *paths],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    index_files = {p for p in (idx.stdout or "").splitlines() if p}

    restored: list[str] = []
    for rel in paths:
        proc = run(
            ["git", "checkout", baseline, "--", rel],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            restored.append(rel)

    # Remove tracked worktree files under ``paths`` that the baseline lacks.
    # ``git checkout <commit> -- <path>`` does NOT delete paths absent from the
    # commit, so a failed run's leftover exp dirs would otherwise survive.
    for rel in index_files - baseline_files:
        target = root / rel
        if target.exists():
            target.unlink()

    return restored


def commit_run(
    root: Path,
    *,
    strategy_id: str,
    status: str,
    cv: float | None,
    extra: list[str] | None = None,
    run=subprocess.run,
) -> str | None:
    """Commit the run's ledger + exp changes (ok or fail). Returns commit msg or None.

    Stages only allow-listed changed paths (never data CSVs / OOF / models / secrets).
    No-op when the root is not a git repo or there is nothing allowed to stage.
    """
    if not is_git_repo(root, run=run):
        return None
    cv_s = f"{cv:.5f}" if cv is not None else "—"
    message = f"loop: RUN {strategy_id} status={status} cv={cv_s}"
    staged = stage_allowed(root, extra, run=run)
    if not staged:
        return None
    proc = run(
        ["git", "commit", "-m", message],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return message
