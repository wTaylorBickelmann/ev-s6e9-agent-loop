"""Planner marker/JSON parse, RESULT lines, and agy envelope helpers."""

from __future__ import annotations

import pytest


from loop.adapters.antigravity import build_agy_cmd, looks_like_credit_failure, parse_agy_envelope
from loop.models import PlannerError
from loop.parse import parse_plan, parse_result_line
from loop.shell import timeout_seconds


def test_parse_markers():
    text = """
<<<ID>>>
s007
<<<PHASE>>>
fe
<<<ONE_LINER>>>
CatBoost + city TE
<<<SPEC>>>
# s007 — CatBoost
hello
<<<END>>>
"""
    plan = parse_plan(text, default_id="s099")
    assert plan.strategy_id == "s007"
    assert plan.phase == "fe"
    assert "CatBoost" in plan.one_liner
    assert "hello" in plan.spec


def test_parse_current_heading_keeps_id():
    spec = "# s001 — LightGBM 5-fold stratified baseline\n\nA baseline.\n"
    plan = parse_plan(spec, default_id="s099")
    assert plan.strategy_id == "s001"
    assert "LightGBM" in plan.one_liner


def test_parse_json_spec():
    plan = parse_plan(
        '{"id":"s002","phase":"stack","one_liner":"ridge stack","spec":"# s002\\n"}',
        default_id="s001",
    )
    assert plan.strategy_id == "s002"
    assert plan.phase == "stack"


def test_result_line():
    row = parse_result_line(
        "noise\nRESULT id=s003 status=ok cv=0.9412 lb=- notes=5-fold mean AUC\n",
        default_id="s000",
    )
    assert row is not None
    assert row.cv == pytest.approx(0.9412)
    assert row.lb is None
    assert row.status == "ok"


def test_agy_envelope_success():
    raw = (
        '{"status":"SUCCESS","response":'
        '"<<<ID>>>\\ns002\\n<<<ONE_LINER>>>\\nx\\n<<<SPEC>>>\\n# s002\\n<<<END>>>"}'
    )
    text = parse_agy_envelope(raw)
    assert "s002" in text


def test_agy_envelope_error():
    with pytest.raises(PlannerError):
        parse_agy_envelope('{"status":"ERROR","error":"authentication required"}')


def test_credit_detector():
    assert looks_like_credit_failure("ERROR: authentication required")
    assert looks_like_credit_failure("RESOURCE_EXHAUSTED quota")
    assert not looks_like_credit_failure("model wrote a strategy")


def test_timeout_parse():
    assert timeout_seconds("10m") == 600
    assert timeout_seconds("90") == 90


def test_agy_model_flag_before_prompt():
    argv = build_agy_cmd(
        {"bin": "agy", "model": "gemini-3.5-flash-high", "effort": "high"},
        "hello",
    )
    assert argv.index("--model") < argv.index("-p")
    assert argv.index("--effort") < argv.index("-p")
    assert argv[-2:] == ["-p", "hello"]
