"""Planner that tries Antigravity, then local DeepSeek on recoverable failure."""

from __future__ import annotations

import sys

from loop.models import Plan, PlannerError


def log(msg: str) -> None:
    """Print a fallback-status line to stderr."""
    print(f"[loop] {msg}", file=sys.stderr)



class FallbackPlanner:
    """Try Antigravity; on any recoverable failure, use local DeepSeek."""

    name = "fallback"

    def __init__(self, primary, fallback):
        """`primary` is usually `agy`; `fallback` is DeepSeek."""
        self.primary = primary
        self.fallback = fallback

    def plan(self, prompt: str) -> Plan:
        """Call primary; on recoverable `PlannerError`, call fallback."""

        try:
            return self.primary.plan(prompt)
        except PlannerError as exc:
            if not exc.recoverable or self.fallback is None:
                raise
            log(f"{self.primary.name} failed ({exc}); falling back to {self.fallback.name}")
            return self.fallback.plan(prompt)
