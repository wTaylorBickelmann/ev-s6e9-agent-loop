"""Append-only CSV scoreboard for every loop run (ok or fail).

After each commit the orchestrator appends one row. Rewind and submit-if-improved
read this file and pick the best CV regardless of ``status=keep`` vs ``fail``.
The CSV and other ledgers are never rewound — they are cumulative memory.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

COLUMNS = (
    "strategy_id",
    "one_liner",
    "commit",
    "commit_msg",
    "cv",
    "lb",
    "status",
    "ts",
)
DEFAULT_REL = Path("ledger/runs_history.csv")
_STRAT_ROW = re.compile(
    r"^\|\s*(s\d+)\s*\|\s*[^|]*\|\s*[^|]*\|\s*(.*?)\s*\|?\s*$",
    re.I,
)


@dataclass
class HistoryRow:
    """One append-only scoreboard row (commit SHA + CV + status)."""

    strategy_id: str
    one_liner: str = ""
    commit: str = ""
    commit_msg: str = ""
    cv: float | None = None
    lb: float | None = None
    status: str = ""
    ts: str = ""


def history_path(root: Path) -> Path:
    """``<root>/ledger/runs_history.csv``."""

    return root / DEFAULT_REL


def utc_now() -> str:
    """UTC timestamp for the ``ts`` column."""

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_metric(value: object) -> float | None:
    """Parse a CSV cell to float; empty / em-dash / n/a → None."""

    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().strip("`")
    if text in {"", "-", "—", "na", "n/a", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    """Write a metric cell (empty when missing — never rewrite later)."""

    if value is None:
        return ""
    return f"{value:.6g}"


def read_rows(path: Path) -> list[HistoryRow]:
    """Load every data row; missing file → ``[]``. Never mutates the file."""

    if not path.is_file():
        return []
    rows: list[HistoryRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            sid = (raw.get("strategy_id") or "").strip()
            if not sid:
                continue
            rows.append(
                HistoryRow(
                    strategy_id=sid,
                    one_liner=(raw.get("one_liner") or "").strip(),
                    commit=(raw.get("commit") or "").strip(),
                    commit_msg=(raw.get("commit_msg") or "").strip(),
                    cv=parse_metric(raw.get("cv")),
                    lb=parse_metric(raw.get("lb")),
                    status=(raw.get("status") or "").strip(),
                    ts=(raw.get("ts") or "").strip(),
                )
            )
    return rows


def append_row(path: Path, row: HistoryRow) -> None:
    """Append one row; write the header only when the file is new. Never rewrite old rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "strategy_id": row.strategy_id,
                "one_liner": row.one_liner,
                "commit": row.commit,
                "commit_msg": row.commit_msg,
                "cv": _fmt(row.cv),
                "lb": _fmt(row.lb),
                "status": row.status,
                "ts": row.ts or utc_now(),
            }
        )


def best_row(
    rows: list[HistoryRow],
    *,
    kind: str = "cv",
    exclude: set[str] | None = None,
    higher_is_better: bool = True,
) -> HistoryRow | None:
    """Row with the best numeric score; skip null/missing. Ties go to the last row."""

    skip = {s.lower() for s in (exclude or set()) if s}
    winner: HistoryRow | None = None
    winner_val: float | None = None
    for row in rows:
        if row.strategy_id.lower() in skip:
            continue
        value = row.cv if kind == "cv" else row.lb
        if value is None:
            continue
        if winner is None or winner_val is None:
            winner, winner_val = row, value
            continue
        if higher_is_better and value >= winner_val:
            winner, winner_val = row, value
        elif not higher_is_better and value <= winner_val:
            winner, winner_val = row, value
    return winner


def one_liner_for(root: Path, strategy_id: str) -> str:
    """One-liner from STRATEGIES.md, else the first CURRENT_STRATEGY heading."""

    strategies = root / "ledger" / "STRATEGIES.md"
    found = ""
    if strategies.is_file():
        for line in strategies.read_text(encoding="utf-8").splitlines():
            match = _STRAT_ROW.match(line.strip())
            if not match:
                continue
            sid = match.group(1).strip()
            if sid.lower() in {"id"}:
                continue
            if sid.lower() == strategy_id.lower():
                found = match.group(2).strip()
    if found:
        return found
    current = root / "ledger" / "CURRENT_STRATEGY.md"
    if current.is_file():
        for line in current.read_text(encoding="utf-8").splitlines():
            text = line.strip().lstrip("#").strip()
            if text:
                return text[:160]
    return ""
