"""Strategy id increment, idempotent appends, and best-CV pick."""

from __future__ import annotations

from pathlib import Path

from loop.ledger import (
    append_result,
    append_strategy,
    best_cv,
    next_strategy_id,
    parse_result_rows,
    update_result_lb,
    write_current,
)
from loop.models import Plan, RunResult


def test_next_id_increments(tmp_path: Path):
    path = tmp_path / "STRATEGIES.md"
    path.write_text(
        "| s001 | 2026-09-12 | baseline | first |\n| s003 | x | fe | skip |\n",
        encoding="utf-8",
    )
    assert next_strategy_id(path) == "s004"


def test_next_id_on_empty(tmp_path: Path):
    assert next_strategy_id(tmp_path / "missing.md") == "s001"


def test_append_and_best(tmp_path: Path):
    strategies = tmp_path / "S.md"
    results = tmp_path / "R.md"
    plan = Plan(strategy_id="s002", one_liner="te + lgbm", spec="# s002\n", phase="fe")
    append_strategy(strategies, plan)
    append_strategy(strategies, plan)  # idempotent
    assert strategies.read_text(encoding="utf-8").count("| s002 |") == 1
    append_result(results, RunResult("s001", "ok", cv=0.90, notes="a"))
    append_result(results, RunResult("s002", "ok", cv=0.94, notes="b"))
    append_result(results, RunResult("s003", "fail", cv=0.95, notes="timeout after train"))
    rows = parse_result_rows(results)
    assert best_cv(rows) == ("s003", 0.95)
    write_current(tmp_path / "C.md", plan)
    assert "# s002" in (tmp_path / "C.md").read_text(encoding="utf-8")
    assert update_result_lb(results, "s002", 0.9415) is True
    updated = parse_result_rows(results)
    s002 = next(r for r in updated if r.strategy_id == "s002")
    assert s002.lb == 0.9415
    assert s002.cv == 0.94
