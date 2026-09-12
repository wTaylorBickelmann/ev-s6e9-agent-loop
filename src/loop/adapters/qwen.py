"""Headless Qwen Code executor: `qwen -p` against a local OpenAI-compatible model."""

from __future__ import annotations

import os
from pathlib import Path

from loop.ledger import parse_result_rows
from loop.models import Plan, RunResult
from loop.parse import parse_result_line
from loop.shell import run_cmd, timeout_seconds


class ExecutorQwenCode:
    """Headless `qwen -p` pointed at a local OpenAI-compatible Qwen endpoint."""

    name = "qwen"

    def __init__(
        self,
        cfg: dict,
        *,
        cwd: Path,
        loop_root: Path,
        extra_dirs: list[Path] | None = None,
    ):
        """`cfg` is `executor.qwen`. `cwd` is where train scripts run."""

        self.cfg = cfg
        self.cwd = cwd
        self.loop_root = loop_root
        self.extra_dirs = extra_dirs or []

    def execute(self, prompt: str, plan: Plan) -> RunResult:
        """Run `qwen -p`, then parse a RESULT line or the new RESULTS.md row."""

        argv = [str(self.cfg.get("bin") or "qwen"), "-p", prompt]
        auth = self.cfg.get("auth_type")
        if auth:
            argv += ["--auth-type", str(auth)]
        model = self.cfg.get("model")
        if model:
            argv += ["--model", str(model)]
        base = self.cfg.get("base_url")
        if base:
            argv += ["--openai-base-url", str(base)]
        key = self.cfg.get("api_key")
        if key:
            argv += ["--openai-api-key", str(key)]
        argv += ["--output-format", str(self.cfg.get("output_format") or "text")]
        if self.cfg.get("yolo", True):
            argv += ["--yolo"]
        wall = self.cfg.get("max_wall_time")
        if wall:
            argv += ["--max-wall-time", str(wall)]
        dirs = [str(p) for p in self.extra_dirs if p.exists()]
        if dirs:
            argv += ["--include-directories", ",".join(dirs)]
        argv += [str(x) for x in (self.cfg.get("extra_args") or [])]

        env = os.environ.copy()
        if base:
            env["OPENAI_BASE_URL"] = str(base)
        if model:
            env["OPENAI_MODEL"] = str(model)
        if key:
            env["OPENAI_API_KEY"] = str(key)

        result = run_cmd(
            argv,
            cwd=self.cwd,
            timeout=timeout_seconds(self.cfg.get("max_wall_time"), 2700.0),
            env=env,
        )
        parsed = parse_result_line(result.stdout, default_id=plan.strategy_id)
        if parsed is None:
            rows = parse_result_rows(self.loop_root / "ledger" / "RESULTS.md")
            match = next((r for r in rows if r.strategy_id == plan.strategy_id), None)
            if match:
                match.phase = plan.phase
                match.source = self.name
                return match
        if parsed:
            parsed.phase = plan.phase
            parsed.source = self.name
            if result.code != 0 and parsed.status == "ok":
                parsed.status = "fail"
                parsed.notes = (parsed.notes + f" ; qwen exit {result.code}").strip(" ;")
            return parsed
        snippet = (result.stderr or result.stdout or "qwen produced no RESULT line").strip()[:200]
        return RunResult(
            strategy_id=plan.strategy_id,
            status="fail",
            notes=f"qwen exit {result.code}: {snippet}",
            phase=plan.phase,
            source=self.name,
        )
