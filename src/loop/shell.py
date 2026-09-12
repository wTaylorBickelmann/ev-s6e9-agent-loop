from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CmdResult:
    code: int
    stdout: str
    stderr: str
    argv: list[str]


def run_cmd(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> CmdResult:
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return CmdResult(127, "", f"executable not found: {argv[0]}", argv)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return CmdResult(124, out, err or "timeout", argv)
    return CmdResult(proc.returncode, proc.stdout or "", proc.stderr or "", argv)


def timeout_seconds(value: str | int | float | None, default: float = 300.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    try:
        if text.endswith("ms"):
            return float(text[:-2]) / 1000.0
        if text.endswith("s"):
            return float(text[:-1])
        if text.endswith("m"):
            return float(text[:-1]) * 60.0
        if text.endswith("h"):
            return float(text[:-1]) * 3600.0
        return float(text)
    except ValueError:
        return default
