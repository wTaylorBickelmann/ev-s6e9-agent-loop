"""Build the planner/executor pair from Settings (live adapters or --dry-run mocks)."""

from __future__ import annotations

import os


from loop.adapters.antigravity import PlannerAntigravity
from loop.adapters.deepseek import PlannerDeepSeek
from loop.adapters.fallback import FallbackPlanner
from loop.adapters.mock import ExecutorMock, PlannerMock
from loop.adapters.qwen import ExecutorQwenCode
from loop.config import Settings


def antigravity_disabled(settings: Settings) -> bool:
    """True when env or loop.yaml says skip `agy` and use DeepSeek only."""

    if os.environ.get("LOOP_DISABLE_ANTIGRAVITY", "").strip() in {"1", "true", "yes"}:
        return True
    return str((settings.raw.get("planner") or {}).get("primary") or "") == "deepseek"


def build_planner(settings: Settings, *, default_id: str, dry_run: bool):
    """Mock, DeepSeek-only, Antigravity, or Antigravity→DeepSeek fallback."""

    if dry_run:
        return PlannerMock(default_id=default_id)
    deepseek = PlannerDeepSeek(settings.section("planner", "deepseek"), default_id=default_id)
    if antigravity_disabled(settings):
        return deepseek
    agy = PlannerAntigravity(
        settings.section("planner", "antigravity"),
        cwd=settings.root,
        default_id=default_id,
    )
    if not (settings.raw.get("planner") or {}).get("fallback_enabled", True):
        return agy
    return FallbackPlanner(agy, deepseek)


def build_executor(settings: Settings, *, dry_run: bool):
    """Mock executor, or Qwen Code pointed at `competition.root`."""

    if dry_run:
        return ExecutorMock()
    comp = settings.competition_root
    cwd = comp if comp and comp.exists() else settings.root
    extra = [settings.root]
    if comp and comp.exists() and comp != settings.root:
        extra.append(comp)
    return ExecutorQwenCode(
        settings.section("executor", "qwen"),
        cwd=cwd,
        loop_root=settings.root,
        extra_dirs=extra,
    )


def competition_root_label(settings: Settings) -> str:
    """Path string for the executor prompt, or `(unset)`."""

    root = settings.competition_root
    if root is None:
        return "(unset)"
    return str(root)
