"""Shared fixtures: copy loop scaffold into a tmp workdir so tests stay isolated."""

from __future__ import annotations

from pathlib import Path

import pytest

from loop.config import load_settings


def _copy_scaffold(src: Path, dest: Path) -> None:
    """Copy config, prompts, ledgers, and planner-always markdown into `dest`."""

    for rel in (
        "config/loop.yaml",
        "config/planner_reads.yaml",
        "prompts/planner.md",
        "prompts/executor.md",
        "ledger/STRATEGIES.md",
        "ledger/RESULTS.md",
        "ledger/CURRENT_STRATEGY.md",
        "ledger/runs_history.csv",
        "CURSOR.md",
        "LEARNINGS.md",
        "STRATEGY.md",
    ):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((src / rel).read_text(encoding="utf-8"), encoding="utf-8")


@pytest.fixture
def repo_root() -> Path:
    """This instance's checkout (parent of `tests/`)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def work(tmp_path: Path, repo_root: Path) -> Path:
    """Temporary loop root with scaffold files; does not mutate the real ledgers."""

    _copy_scaffold(repo_root, tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "ledger" / "runs").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def settings(work: Path):
    """Settings loaded from the temporary work root."""
    return load_settings(work)

