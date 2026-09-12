# Provenance (one-time vendor)

This checkout is a **filled instance**, not a live mirror.

| Source | Ref | What was copied |
|--------|-----|-----------------|
| [kaggle-agent-loop](https://github.com/wTaylorBickelmann/kaggle-agent-loop) | `main` at vendor time | `src/loop/`, adapters, prompts, CLI, dry-run mocks, CI smoke |
| [ev-purchase-kaggle](https://github.com/wTaylorBickelmann/ev-purchase-kaggle) | commit `a4c75ae` | `src/ev_s6e9/`, `scripts/run_exp.py`, strategy memory, `exps/exp0010/` metrics/config/NOTES |

The loop shell repo stays the empty reusable template. This instance does **not**
fetch upstream on kickoff. Later improvements land here via the Antigravity→Qwen
loop (`python -m loop run`) or ordinary PRs.

Not copied: Kaggle CSVs, `oof.csv` (~18 MB), fold models, secrets, Cursor-Fable
`autoloop.py`, planner-fail logs.
