"""Regression tests for loop.gitops: commit-every-run + rewind-to-best-baseline.

These build a real temp git repo so we can exercise `git checkout <baseline> -- exps`
and `git commit` end to end without touching the live instance.
"""

from __future__ import annotations

import subprocess

import pytest

from loop.gitops import (
    best_baseline_commit,
    commit_run,
    is_git_repo,
    rewind_to_best,
)


@pytest.fixture
def gitroot(tmp_path):
    """A real git repo with a best-keep baseline commit and a later broken run."""
    root = tmp_path / "repo"
    root.mkdir()
    for d in ("exps", "exps/exp0010", "src", "ledger", "ledger/runs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "ledger" / "RESULTS.md").write_text("# Results\n", encoding="utf-8")
    (root / "exps" / "exp0010" / "config.json").write_text('{"floor": true}', encoding="utf-8")
    (root / "src" / "model.py").write_text("BEST_CODE\n", encoding="utf-8")

    def _git(*args):
        return subprocess.run(
            ["git", *args], cwd=str(root), text=True, capture_output=True, check=False
        )

    _git("init", "-q")
    _git("add", "-A")
    _git("commit", "-q", "-m", "loop: KEEP exp0010 CV=0.94552")
    # baseline SHA = the best-keep commit
    sha = _git("rev-parse", "HEAD").stdout.strip()

    # Simulate a later (broken) run that touched code + left residue
    (root / "exps" / "exp0042").mkdir()
    (root / "exps" / "exp0042" / "config.json").write_text('{"broken": true}', encoding="utf-8")
    (root / "src" / "model.py").write_text("BROKEN_CODE\n", encoding="utf-8")
    (root / "ledger" / "RESULTS.md").write_text(
        "# Results\n| s042 | fail |\n", encoding="utf-8"
    )
    _git("add", "-A")
    _git("commit", "-q", "-m", "loop: RUN s042 status=fail cv=—")

    return root, sha


def test_is_git_repo(gitroot):
    root, _ = gitroot
    assert is_git_repo(root) is True


def test_not_git_repo_noop(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "file.txt").write_text("x", encoding="utf-8")
    assert is_git_repo(plain) is False
    assert rewind_to_best(plain) is None
    assert commit_run(plain, strategy_id="s999", status="ok", cv=0.5) is None


def test_best_baseline_commit_found(gitroot):
    root, sha = gitroot
    assert best_baseline_commit(root) == sha


def test_rewind_restores_code_to_best_baseline(gitroot):
    root, sha = gitroot
    # Pre-rewind: working tree has the broken run's code
    assert "BROKEN_CODE" in (root / "src" / "model.py").read_text(encoding="utf-8")
    assert (root / "exps" / "exp0042").exists()

    restored = rewind_to_best(root)

    assert restored == ["exps", "src"]
    assert "BEST_CODE" in (root / "src" / "model.py").read_text(encoding="utf-8")
    # The broken run's committed file is reverted out of the working tree
    # (git tracks files, so an empty leftover dir may remain but the file is gone)
    assert not (root / "exps" / "exp0042" / "config.json").exists()
    # The ledger (loop memory) is NOT rewound
    assert "s042" in (root / "ledger" / "RESULTS.md").read_text(encoding="utf-8")


def test_commit_run_commits_ok_and_fail(gitroot):
    root, _ = gitroot
    # ok run
    (root / "exps" / "exp0043").mkdir()
    (root / "exps" / "exp0043" / "config.json").write_text('{"ok": true}', encoding="utf-8")
    msg = commit_run(root, strategy_id="s043", status="ok", cv=0.94577)
    assert msg == "loop: RUN s043 status=ok cv=0.94577"
    log = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=str(root), text=True, capture_output=True
    ).stdout.strip()
    assert log == msg

    # fail run — still committed
    (root / "exps" / "exp0044").mkdir()
    (root / "exps" / "exp0044" / "config.json").write_text('{"fail": true}', encoding="utf-8")
    msg2 = commit_run(root, strategy_id="s044", status="fail", cv=None)
    assert msg2 == "loop: RUN s044 status=fail cv=—"
    log2 = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=str(root), text=True, capture_output=True
    ).stdout.strip()
    assert log2 == msg2


def test_commit_run_never_stages_denied_paths(gitroot):
    root, _ = gitroot
    # A denied path (data CSV + model) must not be staged into the run commit
    (root / "exps" / "exp0045").mkdir()
    (root / "exps" / "exp0045" / "oof.csv").write_text("id,pred\n", encoding="utf-8")
    (root / "exps" / "exp0045" / "model.joblib").write_bytes(b"\x00\x01")
    (root / "exps" / "exp0045" / "config.json").write_text('{"ok": true}', encoding="utf-8")
    msg = commit_run(root, strategy_id="s045", status="ok", cv=0.95)

    assert msg == "loop: RUN s045 status=ok cv=0.95000"
    # Only allow-listed paths are tracked — data CSVs and models are excluded
    files = subprocess.run(
        ["git", "ls-files"], cwd=str(root), text=True, capture_output=True
    ).stdout
    assert "exps/exp0045/config.json" in files
    assert "oof.csv" not in files
    assert "model.joblib" not in files
