"""Regression tests for loop.gitops + history: CSV-best rewind and always-commit.

These build a real temp git repo so we can exercise
`git checkout <baseline> -- exps src/ev_s6e9` and `git commit` end to end
without touching the live instance. Commits set a local user identity so they
succeed on GitHub Actions (no global `user.name` / `user.email`).
"""

from __future__ import annotations

import re
import subprocess

import pytest

from loop.gitops import (
    CommitResult,
    best_baseline_commit,
    commit_run,
    is_git_repo,
    rewind_to_best,
)
from loop.history import (
    HistoryRow,
    append_row,
    best_row,
    history_path,
    one_liner_for,
    read_rows,
)

_SHA = re.compile(r"^[0-9a-f]{7,}$")
_RESTORED = ["exps", "src/ev_s6e9"]


def _git(root, *args):
    """Run a git command in ``root`` and return the CompletedProcess."""

    return subprocess.run(
        ["git", *args], cwd=str(root), text=True, capture_output=True, check=False
    )


def _init_git(root):
    """``git init`` plus a local identity so commits work on Actions (no global user)."""

    proc = _git(root, "init", "-q")
    assert proc.returncode == 0, proc.stderr
    for key, value in (("user.email", "test@example.com"), ("user.name", "test")):
        cfg = _git(root, "config", key, value)
        assert cfg.returncode == 0, cfg.stderr


def _head_sha(root) -> str:
    """``git rev-parse --verify HEAD``; must look like a hex SHA (not the literal HEAD)."""

    proc = _git(root, "rev-parse", "--verify", "HEAD")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    sha = (proc.stdout or "").strip()
    assert _SHA.fullmatch(sha), sha
    return sha


def _commit(root, message: str) -> str:
    """Stage everything, commit, and return the new HEAD SHA. Fail if git refuses."""

    add = _git(root, "add", "-A")
    assert add.returncode == 0, add.stderr
    proc = _git(root, "commit", "-q", "-m", message)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return _head_sha(root)


@pytest.fixture
def gitroot(tmp_path):
    """A real git repo with a KEEP baseline, a later broken run, and a CSV row."""

    root = tmp_path / "repo"
    root.mkdir()
    for d in ("exps", "exps/exp0010", "src/ev_s6e9", "src/loop", "ledger", "ledger/runs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "ledger" / "RESULTS.md").write_text("# Results\n", encoding="utf-8")
    (root / "exps" / "exp0010" / "config.json").write_text('{"floor": true}', encoding="utf-8")
    (root / "src" / "ev_s6e9" / "model.py").write_text("BEST_CODE\n", encoding="utf-8")
    (root / "src" / "loop" / "harness.py").write_text("LOOP_V1\n", encoding="utf-8")

    _init_git(root)
    sha = _commit(root, "loop: KEEP exp0010 CV=0.94552")
    append_row(
        history_path(root),
        HistoryRow(
            strategy_id="s010",
            one_liner="floor keep",
            commit=sha,
            commit_msg="loop: KEEP exp0010 CV=0.94552",
            cv=0.94552,
            status="ok",
        ),
    )

    (root / "exps" / "exp0042").mkdir()
    (root / "exps" / "exp0042" / "config.json").write_text('{"broken": true}', encoding="utf-8")
    (root / "src" / "ev_s6e9" / "model.py").write_text("BROKEN_CODE\n", encoding="utf-8")
    (root / "src" / "loop" / "harness.py").write_text("LOOP_V2\n", encoding="utf-8")
    (root / "ledger" / "RESULTS.md").write_text(
        "# Results\n| s042 | fail |\n", encoding="utf-8"
    )
    _commit(root, "loop: RUN s042 status=fail cv=—")

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


def test_best_baseline_commit_from_csv(gitroot):
    root, sha = gitroot
    assert best_baseline_commit(root) == sha


def test_best_baseline_falls_back_to_keep_grep(tmp_path):
    """Empty CSV → latest ``loop: KEEP`` (migration only)."""

    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x\n", encoding="utf-8")
    _init_git(root)
    sha = _commit(root, "loop: KEEP exp0010 CV=0.94552")
    assert best_baseline_commit(root) == sha


def test_rewind_restores_code_to_best_baseline(gitroot):
    root, sha = gitroot
    assert "BROKEN_CODE" in (root / "src" / "ev_s6e9" / "model.py").read_text(encoding="utf-8")
    assert (root / "exps" / "exp0042").exists()

    restored = rewind_to_best(root)

    assert restored == _RESTORED
    assert "BEST_CODE" in (root / "src" / "ev_s6e9" / "model.py").read_text(encoding="utf-8")
    assert not (root / "exps" / "exp0042" / "config.json").exists()
    # Loop harness + ledger accumulate — they are not part of the rewind tree.
    assert "LOOP_V2" in (root / "src" / "loop" / "harness.py").read_text(encoding="utf-8")
    assert "s042" in (root / "ledger" / "RESULTS.md").read_text(encoding="utf-8")


def test_rewind_uses_fail_row_when_it_has_best_cv(gitroot):
    """A committed ``status=fail`` with a better CV is the rewind baseline."""

    root, keep_sha = gitroot
    (root / "src" / "ev_s6e9" / "model.py").write_text("BETTER_CODE\n", encoding="utf-8")
    (root / "src" / "loop" / "harness.py").write_text("LOOP_V3\n", encoding="utf-8")
    (root / "exps" / "exp0041").mkdir()
    (root / "exps" / "exp0041" / "config.json").write_text('{"cv": 0.94575}', encoding="utf-8")
    fail_sha = _commit(root, "loop: RUN s041 status=fail cv=0.94575")
    assert fail_sha != keep_sha
    append_row(
        history_path(root),
        HistoryRow(
            strategy_id="s041",
            one_liner="timeout after train",
            commit=fail_sha,
            commit_msg="loop: RUN s041 status=fail cv=0.94575",
            cv=0.94575,
            status="fail",
        ),
    )
    (root / "src" / "ev_s6e9" / "model.py").write_text("DIRTY_CODE\n", encoding="utf-8")
    (root / "src" / "loop" / "harness.py").write_text("LOOP_V4\n", encoding="utf-8")

    assert best_baseline_commit(root) == fail_sha
    restored = rewind_to_best(root)
    assert restored == _RESTORED
    assert "BETTER_CODE" in (root / "src" / "ev_s6e9" / "model.py").read_text(encoding="utf-8")
    assert (root / "exps" / "exp0041" / "config.json").exists()
    # harness + ledger / CSV stay on the dirty (latest) working tree
    assert "LOOP_V4" in (root / "src" / "loop" / "harness.py").read_text(encoding="utf-8")
    assert read_rows(history_path(root))[-1].strategy_id == "s041"
    assert "s042" in (root / "ledger" / "RESULTS.md").read_text(encoding="utf-8")


def test_csv_append_is_append_only(tmp_path):
    path = history_path(tmp_path)
    append_row(path, HistoryRow(strategy_id="s001", cv=0.9, status="ok", commit="aaa"))
    append_row(path, HistoryRow(strategy_id="s002", cv=None, status="fail", commit="bbb"))
    rows = read_rows(path)
    assert [r.strategy_id for r in rows] == ["s001", "s002"]
    assert rows[0].cv == pytest.approx(0.9)
    assert rows[1].cv is None
    first = path.read_text(encoding="utf-8")
    append_row(path, HistoryRow(strategy_id="s003", cv=0.91, status="fail", commit="ccc"))
    assert first in path.read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8").count("strategy_id,") == 1


def test_best_row_ignores_null_cv_and_prefers_fail_if_higher():
    rows = [
        HistoryRow(strategy_id="s010", cv=0.94552, status="ok", commit="keep"),
        HistoryRow(strategy_id="s040", cv=None, status="fail", commit="none"),
        HistoryRow(strategy_id="s041", cv=0.94575, status="fail", commit="failbest"),
        HistoryRow(strategy_id="s042", cv=0.94540, status="ok", commit="worse"),
    ]
    winner = best_row(rows)
    assert winner is not None
    assert winner.strategy_id == "s041"
    assert winner.status == "fail"
    assert winner.commit == "failbest"
    assert best_row(rows, exclude={"s041", "exp0041"}).strategy_id == "s010"


def test_one_liner_from_strategies_then_current(tmp_path):
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "STRATEGIES.md").write_text(
        "| id | date | phase | one-liner |\n"
        "| s041 | 2026-09-13 | fe | commute triple TE |\n",
        encoding="utf-8",
    )
    (tmp_path / "ledger" / "CURRENT_STRATEGY.md").write_text(
        "# s044 — unused heading\n", encoding="utf-8"
    )
    assert one_liner_for(tmp_path, "s041") == "commute triple TE"
    assert one_liner_for(tmp_path, "s044") == "s044 — unused heading"


def test_commit_run_commits_ok_and_fail(gitroot):
    root, _ = gitroot
    (root / "exps" / "exp0043").mkdir()
    (root / "exps" / "exp0043" / "config.json").write_text('{"ok": true}', encoding="utf-8")
    msg = commit_run(root, strategy_id="s043", status="ok", cv=0.94577)
    assert isinstance(msg, CommitResult)
    assert msg.message == "loop: RUN s043 status=ok cv=0.94577"
    assert len(msg.sha) >= 7
    log = _git(root, "log", "-1", "--format=%s").stdout.strip()
    assert log == msg.message
    assert _head_sha(root) == msg.sha

    (root / "exps" / "exp0044").mkdir()
    (root / "exps" / "exp0044" / "config.json").write_text('{"fail": true}', encoding="utf-8")
    msg2 = commit_run(root, strategy_id="s044", status="fail", cv=None)
    assert msg2 is not None
    assert msg2.message == "loop: RUN s044 status=fail cv=—"
    log2 = _git(root, "log", "-1", "--format=%s").stdout.strip()
    assert log2 == msg2.message


def test_commit_run_never_stages_denied_paths(gitroot):
    root, _ = gitroot
    (root / "exps" / "exp0045").mkdir()
    (root / "exps" / "exp0045" / "oof.csv").write_text("id,pred\n", encoding="utf-8")
    (root / "exps" / "exp0045" / "model.joblib").write_bytes(b"\x00\x01")
    (root / "exps" / "exp0045" / "config.json").write_text('{"ok": true}', encoding="utf-8")
    msg = commit_run(root, strategy_id="s045", status="ok", cv=0.95)

    assert msg is not None
    assert msg.message == "loop: RUN s045 status=ok cv=0.95000"
    files = _git(root, "ls-files").stdout
    assert "exps/exp0045/config.json" in files
    assert "oof.csv" not in files
    assert "model.joblib" not in files


def test_commit_run_stages_history_csv(gitroot):
    """The scoreboard CSV is allow-listed despite the global ``*.csv`` deny."""

    root, _ = gitroot
    append_row(
        history_path(root),
        HistoryRow(strategy_id="s099", cv=0.1, status="fail", commit="pending"),
    )
    (root / "exps" / "exp0099").mkdir()
    (root / "exps" / "exp0099" / "config.json").write_text("{}", encoding="utf-8")
    msg = commit_run(root, strategy_id="s099", status="fail", cv=0.1)
    assert msg is not None
    files = _git(root, "ls-files").stdout
    assert "ledger/runs_history.csv" in files
