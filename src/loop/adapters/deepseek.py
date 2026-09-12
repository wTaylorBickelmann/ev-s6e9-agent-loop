"""Local DeepSeek planner via OpenAI-compatible chat completions (Ollama / vLLM)."""

from __future__ import annotations

from loop.http_chat import chat_complete

from loop.models import Plan, PlannerError
from loop.parse import parse_plan
from loop.shell import timeout_seconds


class PlannerDeepSeek:
    """Local DeepSeek (or any OpenAI-compatible) chat-completions planner."""

    name = "deepseek"

    def __init__(self, cfg: dict, *, default_id: str):
        """`cfg` is the `planner.deepseek` section (base_url, model, api_key)."""
        self.cfg = cfg
        self.default_id = default_id

    def plan(self, prompt: str) -> Plan:
        """Ask the chat endpoint for a marker-block plan and parse it."""

        base = str(self.cfg.get("base_url") or "").strip()
        model = str(self.cfg.get("model") or "").strip()
        if not base or not model:
            raise PlannerError("DeepSeek base_url/model missing", recoverable=False)
        text = chat_complete(
            base_url=base,
            api_key=str(self.cfg.get("api_key") or "ollama"),
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Kaggle strategist. Reply with the marker block only. "
                        "Do not request files. Do not mention logs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            timeout=timeout_seconds(self.cfg.get("timeout_s"), 180.0),
        )
        if not text.strip():
            raise PlannerError("DeepSeek returned an empty response", recoverable=False)
        plan = parse_plan(text, default_id=self.default_id)
        plan.source = self.name
        return plan
