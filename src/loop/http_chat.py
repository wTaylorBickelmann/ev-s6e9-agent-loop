"""Minimal OpenAI-compatible `/chat/completions` client (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from loop.models import PlannerError


def chat_complete(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 180.0,
) -> str:
    """POST a chat completion and return the first choice's text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise PlannerError(f"chat HTTP {exc.code}: {detail}", recoverable=False) from exc
    except urllib.error.URLError as exc:
        raise PlannerError(f"chat unreachable: {exc.reason}", recoverable=False) from exc
    try:
        return str(body["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise PlannerError(
            "chat response missing choices[0].message.content",
            recoverable=False,
        ) from exc
