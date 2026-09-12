"""Load `.env` and expand `${VAR:-default}` placeholders in loop.yaml."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: Path) -> None:
    """Set missing os.environ keys from a KEY=VALUE dotenv file."""

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def expand(value, environ: dict[str, str] | None = None):
    """Walk strings/dicts/lists and replace `${VAR:-default}` from the environment."""

    env = environ if environ is not None else os.environ

    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            return env.get(match.group(1), match.group(2) if match.group(2) is not None else "")

        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, env) for v in value]
    return value
