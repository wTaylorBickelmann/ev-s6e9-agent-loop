"""Planner / executor Protocols. Adapters implement these; the Loop does not care how."""

from __future__ import annotations

from typing import Protocol

from loop.models import Plan, RunResult


class Planner(Protocol):
    """Turn an assembled prompt into the next experiment `Plan`."""

    name: str

    def plan(self, prompt: str) -> Plan:
        """Return a strategy id, one-liner, phase, and executable spec."""
        ...


class Executor(Protocol):
    """Run a planned experiment and return a compact `RunResult`."""

    name: str

    def execute(self, prompt: str, plan: Plan) -> RunResult:
        """Train / evaluate the spec and report status, CV, and notes."""
        ...

