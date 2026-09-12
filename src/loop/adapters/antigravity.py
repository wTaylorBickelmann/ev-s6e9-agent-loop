from __future__ import annotations

import json
import re
from pathlib import Path

from loop.models import Plan, PlannerError
from loop.parse import parse_plan
from loop.shell import run_cmd, timeout_seconds

# Auth / quota / credit signals that should trip DeepSeek fallback.
CREDIT_RE = re.compile(
    r"authentication required|unauthori[sz]ed|quota|rate.?limit|resource_exhausted|"
    r"credit|billing|429|402|forbidden|not authenticated",
    re.I,
)


def looks_like_credit_failure(text: str) -> bool:
    return bool(CREDIT_RE.search(text or ""))


def parse_agy_envelope(stdout: str) -> str:
    start, end = stdout.find("{"), stdout.rfind("}")
    if start < 0 or end <= start:
        text = stdout.strip()
        if text:
            return text
        raise PlannerError("agy produced empty output")
    try:
        data = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError as exc:
        if stdout.strip():
            return stdout.strip()
        raise PlannerError(f"agy JSON unreadable: {exc}") from exc
    status = str(data.get("status") or "")
    if status and status != "SUCCESS":
        raise PlannerError(data.get("error") or f"agy status={status}")
    structured = data.get("structured_output")
    if isinstance(structured, dict):
        return json.dumps(structured)
    return str(data.get("response") or "")


def build_agy_cmd(cfg: dict, prompt: str) -> list[str]:
    """`--model` / `--effort` must precede `-p` or some agy versions drop them."""
    argv = [str(cfg.get("bin") or "agy")]
    model = cfg.get("model")
    if model:
        argv += ["--model", str(model)]
    effort = cfg.get("effort")
    if effort:
        argv += ["--effort", str(effort)]
    argv += ["--output-format", str(cfg.get("output_format") or "json")]
    timeout_flag = cfg.get("print_timeout")
    if timeout_flag:
        argv += ["--print-timeout", str(timeout_flag)]
    argv += [str(x) for x in (cfg.get("extra_args") or [])]
    argv += ["-p", prompt]
    return argv


class PlannerAntigravity:
    """Headless `agy` wrapper. `--model` / `--effort` always precede `-p`."""

    name = "antigravity"

    def __init__(self, cfg: dict, *, cwd: Path, default_id: str):
        self.cfg = cfg
        self.cwd = cwd
        self.default_id = default_id

    def plan(self, prompt: str) -> Plan:
        argv = build_agy_cmd(self.cfg, prompt)
        result = run_cmd(
            argv,
            cwd=self.cwd,
            timeout=timeout_seconds(self.cfg.get("print_timeout"), 600.0),
        )
        blob = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.code != 0:
            raise PlannerError(_agy_message(result.code, blob))
        text = parse_agy_envelope(result.stdout)
        if not text.strip():
            raise PlannerError("agy returned an empty response")
        plan = parse_plan(text, default_id=self.default_id)
        plan.source = self.name
        return plan


def _agy_message(code: int, blob: str) -> str:
    hint = (
        "agy credit/auth/quota failure"
        if looks_like_credit_failure(blob)
        else "agy failed"
    )
    snippet = blob.strip().replace("\n", " ")[:240]
    return f"{hint} (exit {code}): {snippet}"
