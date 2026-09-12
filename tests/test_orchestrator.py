"""Dry-run loop, execute-once id reuse, whitelist, and CLI `--root` isolation."""

from __future__ import annotations

from loop.cli import main

from loop.ledger import next_strategy_id
from loop.orchestrator import Loop
from loop.parse import parse_plan


def test_dry_run_iteration(settings, work):
    nxt = next_strategy_id(work / "ledger" / "STRATEGIES.md")
    loop = Loop(settings, dry_run=True)
    assert loop.run(1) == 0
    strategies = (work / "ledger" / "STRATEGIES.md").read_text(encoding="utf-8")
    results = (work / "ledger" / "RESULTS.md").read_text(encoding="utf-8")
    current = (work / "ledger" / "CURRENT_STRATEGY.md").read_text(encoding="utf-8")
    assert nxt in strategies
    assert nxt in results
    assert "dry-run mock" in current
    run_json = work / "ledger" / "runs" / f"{nxt}.json"
    assert run_json.is_file()
    assert "0.5" in run_json.read_text(encoding="utf-8")


def test_execute_once_keeps_current_id(settings, work):
    spec = (work / "ledger" / "CURRENT_STRATEGY.md").read_text(encoding="utf-8")
    expected = parse_plan(spec, default_id="s999").strategy_id
    loop = Loop(settings, dry_run=True)
    result = loop.execute_once()
    loop.record(result)
    assert result.strategy_id == expected
    assert f"| {expected} |" in (work / "ledger" / "RESULTS.md").read_text(encoding="utf-8")


def test_show_whitelist_excludes_logs(settings, work):
    (work / "logs" / "s001.log").write_text("HUGE TRACE", encoding="utf-8")
    view = Loop(settings, dry_run=True).show_whitelist()
    assert "HUGE TRACE" not in view.rendered
    assert "ledger/RESULTS.md" in view.listing
    assert "CURSOR.md" in view.listing
    assert "LEARNINGS.md" in view.listing
    assert "STRATEGY.md" in view.listing


def test_cli_dry_run(work, repo_root):
    before = (repo_root / "ledger" / "STRATEGIES.md").read_text(encoding="utf-8")
    assert main(["run", "--iterations", "1", "--dry-run", "--root", str(work)]) == 0
    after = (repo_root / "ledger" / "STRATEGIES.md").read_text(encoding="utf-8")
    assert after == before, "CLI --root must not mutate the real repo ledgers"


def test_cli_show_whitelist(work, capsys):
    assert main(["show-whitelist", "--root", str(work)]) == 0
    out = capsys.readouterr().out
    assert "ledger/STRATEGIES.md" in out
