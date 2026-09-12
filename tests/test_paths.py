"""Settings path resolution for ledger/logs (relative vs absolute)."""

from __future__ import annotations

from pathlib import Path

from loop.config import Settings


def test_logs_dir_relative_under_root(tmp_path: Path):
    s = Settings(root=tmp_path, raw={"paths": {"logs_dir": "logs"}})
    assert s.logs_dir == (tmp_path / "logs").resolve()


def test_logs_dir_absolute_outside_repo(tmp_path: Path):
    outside = (tmp_path / "outside-logs").resolve()
    s = Settings(root=tmp_path / "repo", raw={"paths": {"logs_dir": str(outside)}})
    assert s.logs_dir == outside
    assert not str(s.logs_dir).startswith(str((tmp_path / "repo").resolve()))


def test_logs_dir_expands_user(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = Settings(root=tmp_path / "repo", raw={"paths": {"logs_dir": "~/cache-logs"}})
    assert s.logs_dir == (tmp_path / "cache-logs").resolve()
