from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plan:
    strategy_id: str
    one_liner: str
    spec: str
    phase: str = "baseline"
    source: str = "unknown"


@dataclass
class RunResult:
    strategy_id: str
    status: str  # ok | fail
    cv: float | None = None
    lb: float | None = None
    notes: str = ""
    phase: str = ""
    source: str = "unknown"


@dataclass
class FileSlice:
    rel: str
    text: str
    truncated: bool = False
    missing: bool = False


@dataclass
class AssembledContext:
    listing: list[str] = field(default_factory=list)
    slices: list[FileSlice] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_bytes: int = 0

    def render(self) -> str:
        parts: list[str] = []
        for item in self.slices:
            if item.missing:
                continue
            flag = " (truncated)" if item.truncated else ""
            parts.append(f"### {item.rel}{flag}\n\n{item.text.rstrip()}\n")
        return "\n".join(parts) if parts else "(no whitelist files found)"


class PlannerError(Exception):
    """Planner failed. `recoverable` triggers Antigravity → DeepSeek fallback."""

    def __init__(self, message: str, *, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable
