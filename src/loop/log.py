"""Stderr logger used by the orchestrator and CLI (`[loop] …`)."""

from __future__ import annotations

import sys


def log(msg: str) -> None:
    """Print a one-line status message to stderr."""
    print(f"[loop] {msg}", file=sys.stderr)

