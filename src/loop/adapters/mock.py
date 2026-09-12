"""In-process planner/executor used by `--dry-run` and CI (no CLIs, no GPU)."""

from __future__ import annotations

from loop.models import Plan, RunResult


class PlannerMock:
    """Return a canned spec so CI can exercise ledger writes."""

    name = "mock"

    def __init__(self, *, default_id: str):
        """Remember the next `sNNN` the orchestrator already allocated."""
        self.default_id = default_id

    def plan(self, prompt: str) -> Plan:
        """Build a placeholder Plan for `default_id` (prompt unused)."""

        sid = self.default_id
        one = f"Dry-run mock strategy {sid} (no training)"
        spec = (
            f"# {sid} — dry-run mock\n\n"
            "## Hypothesis\n\n"
            "CI / dry-run placeholder. No model is trained.\n\n"
            "## Changes\n\n"
            "- Do not touch competition data.\n"
            "- Record a synthetic RESULTS row only.\n\n"
            "## Acceptance\n\n"
            "- Orchestrator writes ledgers without calling `agy` or `qwen`.\n"
        )
        _ = prompt  # whitelist prompt is built but unused in the mock
        return Plan(strategy_id=sid, one_liner=one, spec=spec, phase="baseline", source=self.name)


class ExecutorMock:
    """Record a synthetic ok / CV=0.5 result without training."""

    name = "mock"

    def execute(self, prompt: str, plan: Plan) -> RunResult:
        """Return a fake success so the orchestrator can append RESULTS."""

        _ = prompt
        return RunResult(
            strategy_id=plan.strategy_id,
            status="ok",
            cv=0.5,
            lb=None,
            notes="dry-run mock; no training",
            phase=plan.phase,
            source=self.name,
        )
