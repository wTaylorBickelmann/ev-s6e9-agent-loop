from __future__ import annotations

from typing import Protocol

from loop.models import Plan, RunResult


class Planner(Protocol):
    name: str

    def plan(self, prompt: str) -> Plan: ...


class Executor(Protocol):
    name: str

    def execute(self, prompt: str, plan: Plan) -> RunResult: ...
