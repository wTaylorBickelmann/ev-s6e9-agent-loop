"""Fail-fast train_args / schema checks that survive CSV-best rewind."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loop.train_guard import (
    TrainConfigError,
    unknown_schema_tokens,
    unknown_train_flags,
    validate_exp_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_train_flags_rejects_orig():
    assert unknown_train_flags(["--strategy", "deotte", "--freq", "--te"]) == []
    assert unknown_train_flags(["--strategy", "deotte", "--orig"]) == ["--orig"]


def test_validate_rejects_invalid_strategy_choice():
    with pytest.raises(TrainConfigError, match="post-rewind"):
        validate_exp_config({"train_args": ["--strategy", "orig"]})


def test_unknown_schema_tokens_rejects_invented_column():
    bad = unknown_schema_tokens(
        ["fold-safe TE of Years_of_Driving_Experience on Annual_Income_USD"]
    )
    assert bad == ["Years_of_Driving_Experience"]


def test_exp0010_config_passes_guard():
    cfg = json.loads((ROOT / "exps" / "exp0010" / "config.json").read_text(encoding="utf-8"))
    validate_exp_config(cfg)


def test_validate_exp_config_rejects_phantom_flag_and_column():
    with pytest.raises(TrainConfigError, match=r"--orig"):
        validate_exp_config({"train_args": ["--strategy", "deotte", "--orig"]})
    with pytest.raises(TrainConfigError, match="Years_of_Driving_Experience"):
        validate_exp_config(
            {
                "train_args": ["--strategy", "deotte", "--freq", "--te"],
                "hypothesis": "Add Years_of_Driving_Experience interaction",
            }
        )


def test_committed_exp_configs_pass_guard():
    """Existing scored configs must still train; the guard is fail-fast only for phantoms."""

    for path in sorted((ROOT / "exps").glob("exp*/config.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        validate_exp_config(cfg)


def test_run_exp_invokes_guard_before_train():
    text = (ROOT / "scripts" / "run_exp.py").read_text(encoding="utf-8")
    assert "validate_exp_config" in text
    assert "TrainConfigError" in text
