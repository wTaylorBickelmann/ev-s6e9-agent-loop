from __future__ import annotations

import sys


def log(msg: str) -> None:
    print(f"[loop] {msg}", file=sys.stderr)
