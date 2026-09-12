"""This instance: planner sees ledgers + EV entrypoints, never OOF/CSVs."""

from __future__ import annotations

from loop.cli import main
from loop.config import load_settings
from loop.ledger import next_strategy_id
from loop.orchestrator import Loop


def test_seeded_next_id_is_after_exp0018(repo_root):
    assert next_strategy_id(repo_root / "ledger" / "STRATEGIES.md") == "s019"


def test_repo_whitelist_includes_floor_and_denies_dumps(repo_root):
    settings = load_settings(repo_root)
    view = Loop(settings, dry_run=True).show_whitelist()
    joined = "\n".join(view.listing)
    assert "LEARNINGS.md" in joined
    assert "STRATEGY.md" in joined
    assert "src/ev_s6e9/features.py" in joined
    assert "src/ev_s6e9/deotte.py" in joined
    assert "exps/exp0010/NOTES.md" in joined
    assert "oof.csv" not in view.rendered.lower()
    notes = (repo_root / "exps" / "exp0010" / "NOTES.md").read_text(encoding="utf-8")
    assert "OOF not committed" in notes


def test_cli_show_whitelist_repo(repo_root, capsys):
    assert main(["show-whitelist", "--root", str(repo_root)]) == 0
    out = capsys.readouterr().out
    assert "LEARNINGS.md" in out
    assert "oof.csv" not in out
