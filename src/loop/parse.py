from __future__ import annotations

import json
import re

from loop.models import Plan, RunResult

_RESULT = re.compile(
    r"RESULT\s+id=(?P<id>\S+)\s+status=(?P<status>\S+)\s+"
    r"cv=(?P<cv>\S+)\s+lb=(?P<lb>\S+)\s+notes=(?P<notes>.*)$",
    re.I | re.M,
)


def parse_plan(text: str, *, default_id: str) -> Plan:
    data = _extract_json_object(text)
    if data and ("spec" in data or "one_liner" in data or "id" in data):
        spec = str(data.get("spec") or data.get("body") or "").strip()
        sid = str(data.get("id") or default_id).strip() or default_id
        return Plan(
            strategy_id=_clean_id(sid, default_id),
            one_liner=str(
                data.get("one_liner") or data.get("title") or _first_heading(spec) or sid
            ),
            spec=spec or _fallback_spec(sid, str(data.get("one_liner") or "")),
            phase=_clean_phase(str(data.get("phase") or "baseline")),
        )
    markers = _markers(text)
    if markers.get("SPEC") or markers.get("ONE_LINER"):
        sid = _clean_id(markers.get("ID") or default_id, default_id)
        spec = (markers.get("SPEC") or "").strip()
        one = (markers.get("ONE_LINER") or _first_heading(spec) or sid).strip()
        return Plan(
            strategy_id=sid,
            one_liner=one.splitlines()[0][:160],
            spec=spec or _fallback_spec(sid, one),
            phase=_clean_phase(markers.get("PHASE") or "baseline"),
        )
    spec = text.strip()
    heading = _first_heading(spec)
    return Plan(
        strategy_id=_clean_id(heading or spec or default_id, default_id),
        one_liner=heading or default_id,
        spec=spec or _fallback_spec(default_id, "planner returned empty spec"),
        phase=_clean_phase(spec),
    )


def parse_result_line(text: str, *, default_id: str) -> RunResult | None:
    match = _RESULT.search(text or "")
    if not match:
        return None
    return RunResult(
        strategy_id=_clean_id(match.group("id"), default_id),
        status="ok" if match.group("status").lower() in {"ok", "pass", "success"} else "fail",
        cv=_num(match.group("cv")),
        lb=_num(match.group("lb")),
        notes=match.group("notes").strip(),
    )


def _markers(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    pattern = re.compile(r"<<<([A-Z_]+)>>>\s*(.*?)\s*(?=<<<|$)", re.S)
    for key, value in pattern.findall(text):
        out[key] = value.strip()
    return out


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = fence.group(1) if fence else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            blob = text[start : end + 1]
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clean_id(value: str, default: str) -> str:
    match = re.search(r"s\d+", value, re.I)
    return match.group(0).lower() if match else default


def _clean_phase(value: str) -> str:
    low = value.strip().lower()
    for name in ("eda", "baseline", "fe", "stack"):
        if re.search(rf"(?<![a-z]){name}(?![a-z])", low):
            return name
    return "baseline"


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:160]
    return ""


def _fallback_spec(sid: str, one_liner: str) -> str:
    return f"# {sid}\n\n## Hypothesis\n\n{one_liner or 'See planner output.'}\n"


def _num(value: str) -> float | None:
    text = value.strip()
    if text in {"", "-", "—", "na", "n/a", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
