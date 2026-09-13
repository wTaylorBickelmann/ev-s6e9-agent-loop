"""Executor timeout recovery: a finished training run (scored metrics.json) must not
be recorded as a fail just because the qwen process timed out before emitting its
RESULT line."""

from __future__ import annotations

import json

from loop.adapters.qwen import ExecutorQwenCode
from loop.models import Plan
from loop.shell import CmdResult

import loop.adapters.qwen as qwen_mod

# s999 is absent from the scaffold RESULTS.md so the RESULTS.md short-circuit
# doesn't fire; it maps to exps/exp0999.
CFG = {"bin": "qwen", "model": "qwen3.8:27b-q4_K_M", "max_wall_time": "45m"}
SID = "s999"
EXP = "exp0999"


def _executor(work) -> ExecutorQwenCode:
    return ExecutorQwenCode(CFG, cwd=work, loop_root=work)


def _plan() -> Plan:
    return Plan(strategy_id=SID, one_liner="recover", spec="", phase="stack")


def _timeout_result() -> CmdResult:
    return CmdResult(124, "", "timeout", ["qwen"])


def _write_scored_metrics(work, cv: float) -> None:
    exp_dir = work / "exps" / EXP
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics.json").write_text(
        json.dumps({"id": EXP, "cv_mean": cv, "status": "scored"}),
        encoding="utf-8",
    )


def test_timeout_with_scored_metrics_is_ok(monkeypatch, work):
    _write_scored_metrics(work, 0.94558)
    monkeypatch.setattr(qwen_mod, "run_cmd", lambda *a, **k: _timeout_result())

    result = _executor(work).execute("prompt", _plan())

    assert result.status == "ok"
    assert result.cv == 0.94558
    assert result.strategy_id == SID
    assert result.phase == "stack"
    assert result.source == "qwen"
    assert "recovered" in result.notes


def test_timeout_without_metrics_is_fail(monkeypatch, work):
    # No exps/exp0999/metrics.json — genuine failure.
    monkeypatch.setattr(qwen_mod, "run_cmd", lambda *a, **k: _timeout_result())

    result = _executor(work).execute("prompt", _plan())

    assert result.status == "fail"
    assert result.cv is None


def test_timeout_with_unscored_metrics_is_fail(monkeypatch, work):
    exp_dir = work / "exps" / EXP
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics.json").write_text(
        json.dumps({"id": EXP, "status": "failed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(qwen_mod, "run_cmd", lambda *a, **k: _timeout_result())

    result = _executor(work).execute("prompt", _plan())

    assert result.status == "fail"
    assert result.cv is None
