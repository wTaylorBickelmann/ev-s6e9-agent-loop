"""Fail-fast checks for exp configs against the post-rewind train CLI + schema.

Lives in ``src/loop/`` so CSV-best rewind of ``exps/`` + ``src/ev_s6e9/`` cannot
wipe it. ``scripts/run_exp.py`` calls this before launching train so unknown
flags (e.g. ``--orig`` after rewind) and invented column names fail immediately
instead of dying mid-run.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from contextlib import redirect_stderr

from ev_s6e9.__main__ import _parser
from ev_s6e9.schema import TRAIN_COLS

# Competition raw cols are Pascal_Snake. Derived helpers are snake_case and
# would not match this pattern even if omitted from the allow-set.
_COL_LIKE = re.compile(r"^[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")

_KNOWN_DERIVED = frozenset(
    {
        "worry_score",
        "chargers_total",
        "income_x_subsidy",
        "concern_x_subsidy",
        "no_home_charge_x_high_anxiety",
        "recipe_score",
        "charging_total",
        "Annual_Income_USD_cnt",
        "Daily_Commute_km_cnt",
        "Annual_Income_USD_te",
        "Annual_Income_USD_te_m5",
        "Annual_Income_USD_te_m2",
        "Daily_Commute_km_te",
        "Daily_Commute_km_te_m5",
        "Daily_Commute_km_te_m2",
    }
)


class TrainConfigError(ValueError):
    """Invalid train_args or invented column names vs the post-rewind tree."""


def known_input_cols() -> set[str]:
    """train.csv header plus documented derived / TE / freq names."""

    return set(TRAIN_COLS) | set(_KNOWN_DERIVED)


def _train_option_strings() -> set[str]:
    """Exact ``python -m ev_s6e9 train`` flag names (no argparse prefix matching)."""

    action = next(
        a for a in _parser()._actions if getattr(a, "choices", None) and "train" in a.choices
    )
    out: set[str] = set()
    for opt in action.choices["train"]._actions:
        out.update(opt.option_strings)
    return out


def unknown_train_flags(train_args: list[str]) -> list[str]:
    """CLI flags in ``train_args`` that the current ``python -m ev_s6e9 train`` rejects.

    Matches option strings exactly so ``--orig`` is not silently treated as
    an abbreviation of ``--override``.
    """

    known = _train_option_strings()
    unknown: list[str] = []
    for tok in train_args or []:
        text = str(tok)
        if not text.startswith("-"):
            continue
        key = text.split("=", 1)[0]
        if key not in known:
            unknown.append(tok)
    return unknown


def parse_train_args_or_raise(train_args: list[str]) -> None:
    """Run argparse on known flags; raise if values are invalid (e.g. bad --strategy)."""

    argv = ["train", *list(train_args or [])]
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            _parser().parse_args(argv)
    except SystemExit as exc:
        msg = (err.getvalue() or str(exc) or "invalid train_args").strip()
        raise TrainConfigError(
            "invalid train_args for the current (post-rewind) "
            f"python -m ev_s6e9 train CLI: {msg}. "
            "Do not rely on leftover src/ev_s6e9 edits; implement the flag in "
            "this execute turn before train, or use existing flags."
        ) from exc


def unknown_schema_tokens(texts: Iterable[str], *, known: set[str] | None = None) -> list[str]:
    """Pascal_Snake tokens in ``texts`` that are not train.csv / known derived cols."""

    allowed = known if known is not None else known_input_cols()
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(text)):
            if tok in seen or not _COL_LIKE.match(tok):
                continue
            if tok not in allowed:
                seen.add(tok)
                found.append(tok)
    return found


def validate_exp_config(cfg: dict) -> None:
    """Raise ``TrainConfigError`` on unknown CLI flags or invented column names."""

    train_args = list(cfg.get("train_args") or [])
    flags = unknown_train_flags(train_args)
    if flags:
        raise TrainConfigError(
            f"unknown train flags {flags}: not on the current (post-rewind) "
            "python -m ev_s6e9 train CLI. Do not rely on leftover src/ev_s6e9 "
            "edits; implement the flag in this execute turn before train, or "
            "use existing flags."
        )
    parse_train_args_or_raise(train_args)
    blobs = [
        *train_args,
        str(cfg.get("hypothesis") or ""),
        str(cfg.get("note") or ""),
        str(cfg.get("title") or ""),
    ]
    bad = unknown_schema_tokens(blobs)
    if bad:
        raise TrainConfigError(
            f"unknown train.csv columns {bad}; schema is {list(TRAIN_COLS)}. "
            "Do not invent fields (e.g. Years_of_Driving_Experience)."
        )
