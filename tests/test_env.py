"""`${VAR:-default}` expansion used by loop.yaml."""

from __future__ import annotations

from loop.env import expand


def test_expand_default_and_override():
    env = {"AGY_MODEL": "gemini-3.5-flash-high"}
    assert expand("${AGY_MODEL:-x}", env) == "gemini-3.5-flash-high"
    assert expand("${MISSING:-deepseek-r1:32b}", env) == "deepseek-r1:32b"
    assert expand({"bin": "${AGY_BIN:-agy}"}, env)["bin"] == "agy"
