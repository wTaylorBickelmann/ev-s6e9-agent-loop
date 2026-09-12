from __future__ import annotations

from loop.adapters.fallback import FallbackPlanner
from loop.models import Plan, PlannerError


class _Boom:
    name = "antigravity"

    def plan(self, prompt: str) -> Plan:
        raise PlannerError("authentication required")


class _Ok:
    name = "deepseek"

    def plan(self, prompt: str) -> Plan:
        return Plan(strategy_id="s009", one_liner=prompt[:20], spec="# s009\n", source=self.name)


def test_fallback_on_agy_failure():
    planner = FallbackPlanner(_Boom(), _Ok())
    plan = planner.plan("hello")
    assert plan.source == "deepseek"
    assert plan.strategy_id == "s009"


def test_no_fallback_when_not_recoverable():
    class Fatal:
        name = "antigravity"

        def plan(self, prompt: str) -> Plan:
            raise PlannerError("logic bug", recoverable=False)

    planner = FallbackPlanner(Fatal(), _Ok())
    try:
        planner.plan("x")
    except PlannerError as exc:
        assert not exc.recoverable
    else:
        raise AssertionError("expected PlannerError")
