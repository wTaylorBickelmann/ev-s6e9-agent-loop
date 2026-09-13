#!/usr/bin/env python3
"""Submit to Kaggle and git-commit when the run beats the CSV-best CV.

Usage:
  python scripts/submit_if_improved.py exp0019 --dry-run
  python scripts/submit_if_improved.py exp0019
  python scripts/submit_if_improved.py exp0019 --gate lb --push
  python -m loop submit-if-improved exp0019 --dry-run
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loop.submit_if_improved import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
