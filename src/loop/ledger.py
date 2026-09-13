"""Compact ledger I/O: next id, append STRATEGIES/RESULTS, rewrite CURRENT_STRATEGY."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from loop.models import Plan, RunResult

_ID = re.compile(r"\bs(\d+)\b", re.I)
_ROW = re.compile(
    r"^\|\s*(s\d+)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(.*?)\s*\|?\s*$",
    re.I,
)


def next_strategy_id(strategies_md: Path) -> str:
    """Next `sNNN` after the highest id already in STRATEGIES.md (`s001` if empty)."""

    n = 0
    if strategies_md.is_file():
        for match in _ID.finditer(strategies_md.read_text(encoding="utf-8")):
            n = max(n, int(match.group(1)))
    return f"s{n + 1:03d}"


def parse_result_rows(results_md: Path) -> list[RunResult]:
    """Parse RESULTS.md table rows into RunResult (skip header / separator)."""

    if not results_md.is_file():
        return []
    rows: list[RunResult] = []
    for line in results_md.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line.strip())
        if not match or match.group(1).lower() == "id":
            continue
        status = match.group(2).strip()
        if status in {"status", "----", "---"}:
            continue
        rows.append(
            RunResult(
                strategy_id=match.group(1).strip(),
                status=status,
                cv=_maybe_float(match.group(3)),
                lb=_maybe_float(match.group(4)),
                notes=match.group(5).strip().strip("|").strip(),
            )
        )
    return rows


def best_cv(
    results: list[RunResult], *, higher_is_better: bool = True
) -> tuple[str, float] | None:
    """Winning `(strategy_id, cv)` among scored rows (any status), or None."""

    scored = [r for r in results if r.cv is not None]
    if not scored:
        return None
    winner = max(scored, key=lambda r: r.cv or 0.0) if higher_is_better else min(
        scored, key=lambda r: r.cv or 0.0
    )
    assert winner.cv is not None
    return winner.strategy_id, winner.cv


def phase_coverage(strategies_md: Path) -> str:
    """Counts of eda/baseline/fe/stack rows for the planner prompt."""

    counts = {"eda": 0, "baseline": 0, "fe": 0, "stack": 0}
    if strategies_md.is_file():
        for line in strategies_md.read_text(encoding="utf-8").splitlines():
            low = line.lower()
            for phase in counts:
                if f"| {phase} |" in low or f"|{phase}|" in low:
                    counts[phase] += 1
    return " ".join(f"{k}={v}" for k, v in counts.items())


def has_result(results_md: Path, strategy_id: str) -> bool:
    """True if RESULTS.md already has a row for this id."""

    return any(r.strategy_id == strategy_id for r in parse_result_rows(results_md))


def has_strategy(strategies_md: Path, strategy_id: str) -> bool:
    """True if STRATEGIES.md already has a table row for this id."""

    if not strategies_md.is_file():
        return False
    text = strategies_md.read_text(encoding="utf-8")
    return bool(re.search(rf"\|\s*{re.escape(strategy_id)}\s*\|", text))


def write_current(path: Path, plan: Plan) -> None:
    """Overwrite CURRENT_STRATEGY.md with the full spec."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.spec.rstrip() + "\n", encoding="utf-8")


def append_strategy(path: Path, plan: Plan, when: date | None = None) -> None:
    """Append one STRATEGIES.md row; no-op if the id is already present."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(_STRATEGIES_HEADER, encoding="utf-8")
    if has_strategy(path, plan.strategy_id):
        return
    day = (when or date.today()).isoformat()
    line = f"| {plan.strategy_id} | {day} | {plan.phase} | {plan.one_liner} |\n"
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + line, encoding="utf-8")


def append_result(path: Path, result: RunResult) -> None:
    """Append one RESULTS.md row; no-op if the id is already present."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(_RESULTS_HEADER, encoding="utf-8")
    if has_result(path, result.strategy_id):
        return
    line = (
        f"| {result.strategy_id} | {result.status} | {_fmt(result.cv)} | "
        f"{_fmt(result.lb)} | {result.notes or '—'} |\n"
    )
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + line, encoding="utf-8")


def update_result_lb(path: Path, strategy_id: str, lb: float) -> bool:
    """Overwrite the LB cell on an existing RESULTS.md row. Return True if rewritten."""

    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    changed = False
    for line in lines:
        match = _ROW.match(line.strip())
        if (
            match
            and match.group(1).strip() == strategy_id
            and match.group(1).lower() != "id"
            and match.group(2).strip().lower() not in {"status", "----", "---"}
        ):
            status = match.group(2).strip()
            cv = match.group(3).strip()
            notes = match.group(5).strip().strip("|").strip()
            out.append(f"| {strategy_id} | {status} | {cv} | {_fmt(lb)} | {notes or '—'} |")
            changed = True
        else:
            out.append(line)
    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def write_run_json(path: Path, result: RunResult) -> None:
    """Write metrics-only JSON under ledger/runs/ (never planner-visible)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": result.strategy_id,
        "status": result.status,
        "cv": result.cv,
        "lb": result.lb,
        "notes": result.notes,
        "phase": result.phase,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def current_spec(path: Path) -> str:
    """Full CURRENT_STRATEGY.md text, or empty if the file is missing."""

    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _maybe_float(value: str) -> float | None:
    """Parse a table cell to float; em-dash / n/a / junk → None."""

    text = value.strip().strip("`")
    if text in {"", "-", "—", "na", "n/a", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    """Format a metric cell (`—` when missing)."""

    if value is None:
        return "—"
    return f"{value:.6g}"


_STRATEGIES_HEADER = """# Strategies

Append-only catalog. One row per planned idea. Never paste logs.

| id | date | phase | one-liner |
|----|------|-------|-----------|
"""

_RESULTS_HEADER = """# Results

One row per executed run. Metrics only. Full traces live under LOOP_LOGS_DIR (`<id>.log`).

| id | status | cv | lb | notes |
|----|--------|----|----|-------|
"""
