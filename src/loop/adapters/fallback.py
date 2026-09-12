from __future__ import annotations

import sys

from loop.models import Plan, PlannerError


def log(msg: str) -> None:
    print(f"[loop] {msg}", file=sys.stderr)


class FallbackPlanner:
    """Try Antigravity; on any recoverable failure, use local DeepSeek."""

    name = "fallback"

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def plan(self, prompt: str) -> Plan:
        try:
            return self.primary.plan(prompt)
        except PlannerError as exc:
            if not exc.recoverable or self.fallback is None:
                raise
            log(f"{self.primary.name} failed ({exc}); falling back to {self.fallback.name}")
            return self.fallback.plan(prompt)
