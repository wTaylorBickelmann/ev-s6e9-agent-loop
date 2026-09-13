"""Git helpers: commit every run and rewind train/FE code to the CSV-best CV.

Wired into ``Loop.run`` (live mode only; skipped when the root is not a git repo
so dry-run tests on temp dirs stay green):

1. ``rewind_to_best`` — at the START of every iteration, restore only the
   training / feature-engineering / submission-build tree (``exps/`` and
   ``src/ev_s6e9/``) to the commit on the CSV row with the best CV (higher ROC
   AUC; any status, including ``fail`` with a valid CV). Timeout-after-success
   and kill-vs-floor rows therefore become the next baseline. The AI loop
   harness (``src/loop/``, prompts, ``config/loop.yaml``), ``ledger/``
   (including ``runs_history.csv``), RESULTS, and root memory markdown are
   never rewound — they accumulate.

2. ``commit_run`` — after ``record``, git-commit every allow-listed changed path
   (same deny-list as submit-if-improved) with ``loop: RUN <id> status=… cv=…``.
   Runs for both ok and fail. Returns the new SHA so the orchestrator can append
   the CSV row.

KEEP-grep is only a migration fallback when the CSV has no scored commit yet.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from loop.history import best_row, history_path, read_rows
from loop.submit_if_improved import stage_allowed

# Train / FE / submission-build only. Never the loop harness or ledgers.
_REWIND_PATHS = ("exps", "src/ev_s6e9")


@dataclass(frozen=True)
class CommitResult:
    """SHA and subject of a ``loop: RUN`` commit."""

    sha: str
    message: str


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


def head_sha(root: Path, *, run=subprocess.run) -> str | None:
    """Current ``HEAD`` SHA, or None if git fails."""

    proc = run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    sha = (proc.stdout or "").strip()
    return sha or None if proc.returncode == 0 else None


def _object_exists(root: Path, sha: str, *, run=subprocess.run) -> bool:
    """True when ``sha`` names an object in this repo."""

    proc = run(
        ["git", "cat-file", "-t", sha],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _keep_grep_commit(root: Path, *, run=subprocess.run) -> str | None:
    """Latest ``loop: KEEP`` commit SHA (migration fallback only)."""

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


def best_baseline_commit(root: Path, *, run=subprocess.run) -> str | None:
    """SHA of the CSV row with the best CV; KEEP-grep only if the CSV has none."""

    row = best_row(read_rows(history_path(root)))
    if row and row.commit and _object_exists(root, row.commit, run=run):
        return row.commit
    return _keep_grep_commit(root, run=run)


def rewind_to_best(
    root: Path,
    *,
    paths: tuple[str, ...] = _REWIND_PATHS,
    run=subprocess.run,
) -> list[str] | None:
    """Reset ``paths`` to the CSV-best CV commit. Returns restored paths or None.

    No-op when the root is not a git repo or there is no baseline yet. Restores
    only ``exps/`` and ``src/ev_s6e9/`` so the next train starts from the best
    scored tree (even if that run was ``status=fail``). ``src/loop/`` and
    ledgers keep accumulating.

    ``git checkout <baseline> -- <path>`` does not delete worktree files absent
    from the baseline, so leftover exp dirs from a failed run are unlinked.
    """

    if not is_git_repo(root, run=run):
        return None
    baseline = best_baseline_commit(root, run=run)
    if baseline is None:
        return None

    ls = run(
        ["git", "ls-tree", "-r", "--name-only", baseline, "--", *paths],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    baseline_files = {p for p in (ls.stdout or "").splitlines() if p}

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
) -> CommitResult | None:
    """Commit the run's ledger + exp changes (ok or fail). Returns SHA + message.

    Stages only allow-listed paths (never data CSVs / OOF / models / secrets).
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
    sha = head_sha(root, run=run)
    if not sha:
        return None
    return CommitResult(sha=sha, message=message)
