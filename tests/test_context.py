from __future__ import annotations

from pathlib import Path

from loop.context import assemble, is_denied


def test_denied_prefixes():
    assert is_denied("logs/s001.log", [])
    assert is_denied("ledger/runs/s001.json", ["ledger/runs/"])
    assert is_denied("data/train.csv", ["data/"])
    assert is_denied("exps/exp0010/oof.csv", [])
    assert is_denied("outputs/submission.csv", [])
    assert not is_denied("ledger/RESULTS.md", ["logs/", "ledger/runs/"])
    assert not is_denied("exps/exp0010/NOTES.md", [])


def test_deny_wins_even_if_listed(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "secret.log").write_text("LEAK", encoding="utf-8")
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "RESULTS.md").write_text("ok", encoding="utf-8")
    reads = tmp_path / "config" / "planner_reads.yaml"
    reads.write_text(
        "always:\n  - logs/secret.log\n  - ledger/RESULTS.md\ndeny:\n  - logs/\n",
        encoding="utf-8",
    )
    ctx = assemble(loop_root=tmp_path, reads_path=reads)
    rendered = ctx.render()
    assert "LEAK" not in rendered
    assert "ledger/RESULTS.md" in rendered
    assert any("denied" in s for s in ctx.skipped)


def test_optional_missing_is_skipped(tmp_path: Path):
    (tmp_path / "config").mkdir()
    reads = tmp_path / "config" / "planner_reads.yaml"
    reads.write_text("always: []\noptional:\n  - src/train.py\n", encoding="utf-8")
    ctx = assemble(loop_root=tmp_path, reads_path=reads)
    assert ctx.slices == []
    assert any("missing" in s for s in ctx.skipped)


def test_byte_cap_truncates(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "big.md").write_text("x" * 5000, encoding="utf-8")
    reads = tmp_path / "config" / "planner_reads.yaml"
    reads.write_text(
        "always:\n  - big.md\nmax_bytes_per_file: 100\nmax_total_bytes: 1000\n",
        encoding="utf-8",
    )
    ctx = assemble(loop_root=tmp_path, reads_path=reads)
    assert ctx.slices[0].truncated
    assert "truncated" in ctx.slices[0].text
    assert ctx.total_bytes <= 100
