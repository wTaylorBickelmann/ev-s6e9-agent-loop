"""Dry-run loop, execute-once id reuse, whitelist, and CLI `--root` isolation."""

from __future__ import annotations

from loop.cli import main
from loop.history import history_path, read_rows
from loop.ledger import next_strategy_id
from loop.models import RunResult
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
    hist = read_rows(history_path(work))
    assert hist[-1].strategy_id == nxt
    assert hist[-1].cv == 0.5
    assert hist[-1].status == "ok"
    assert "Dry-run mock" in hist[-1].one_liner


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
    assert "ledger/runs_history.csv" in view.listing
    assert "CURSOR.md" in view.listing
    assert "LEARNINGS.md" in view.listing
    assert "STRATEGY.md" in view.listing


def test_cli_dry_run(work, repo_root):
    before = (repo_root / "ledger" / "STRATEGIES.md").read_text(encoding="utf-8")
    assert main(["run", "--iterations", "1", "--dry-run", "--root", str(work)]) == 0
    after = (repo_root / "ledger" / "STRATEGIES.md").read_text(encoding="utf-8")
    assert after == before, "CLI --root must not mutate the real repo ledgers"


def test_finish_run_submits_fail_with_valid_cv(settings, work, monkeypatch):
    """Live finish_run calls submit-if-improved even when status=fail but CV exists."""

    exp = work / "exps" / "exp0041"
    exp.mkdir(parents=True)
    (exp / "config.json").write_text("{}", encoding="utf-8")
    called: dict = {}
    monkeypatch.setattr("loop.orchestrator.gitops.commit_run", lambda *_a, **_k: None)

    def fake_submit(_root, exp_id, **kwargs):
        called["exp"] = exp_id
        called["eps"] = kwargs.get("eps")
        return 0

    monkeypatch.setattr("loop.orchestrator.submit_if_improved", fake_submit)
    Loop(settings, dry_run=False).finish_run(RunResult("s041", "fail", cv=0.94575))
    assert called["exp"] == "exp0041"
    assert called["eps"] == settings.submit_eps
    row = read_rows(history_path(work))[-1]
    assert row.strategy_id == "s041"
    assert row.status == "fail"
    assert row.cv == 0.94575


def test_finish_run_skips_submit_without_cv(settings, work, monkeypatch):
    monkeypatch.setattr("loop.orchestrator.gitops.commit_run", lambda *_a, **_k: None)

    def _boom(*_a, **_k):
        raise AssertionError("submit must not run without a CV")

    monkeypatch.setattr("loop.orchestrator.submit_if_improved", _boom)
    Loop(settings, dry_run=False).finish_run(RunResult("s099", "fail", cv=None))
    assert read_rows(history_path(work))[-1].strategy_id == "s099"


def test_cli_show_whitelist(work, capsys):
    assert main(["show-whitelist", "--root", str(work)]) == 0
    out = capsys.readouterr().out
    assert "ledger/STRATEGIES.md" in out
