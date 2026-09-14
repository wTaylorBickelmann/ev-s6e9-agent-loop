# Coding conventions (this S6E9 instance)

Token thrift is a feature. Models that wander into `logs/` burn the budget.
This repo is a **filled instance** of [kaggle-agent-loop](https://github.com/wTaylorBickelmann/kaggle-agent-loop):
the loop harness lives in `src/loop/`; the competition library lives in `src/ev_s6e9/`.

```
rewind exps/+src/ev_s6e9/ to CSV-best CV commit (never src/loop/)
        |
        v
planner (agy -> DeepSeek) --> CURRENT_STRATEGY + STRATEGIES.md
        |
        v
executor (qwen) --> copy last best -> exps/expNNNN
        |           scripts/run_exp.py / python -m ev_s6e9 train
        v
RESULTS.md + ledger/runs_history.csv  (+ runs/<id>.json; logs on disk)
        |
        v
always git-commit (ok or fail) --> submit-if-improved if CV beats CSV-best
        |
        v
next plan (whitelist only — never logs/ or oof.csv)
```

Full boxes: `README.md` and `docs/architecture.txt`.

## Planner whitelist

The planner may see **only** paths listed in `config/planner_reads.yaml`.
The orchestrator inlines those files. Do not add `logs/`, `ledger/runs/`,
data dumps, `oof.csv`, notebooks, or `EXPERIMENTS.md` to that list.

## Ledgers stay small

| File | Role | Mutation |
|------|------|----------|
| `ledger/STRATEGIES.md` | id / date / phase / one-liner | append one row per plan |
| `ledger/RESULTS.md` | id / status / cv / lb / notes | append one row per run |
| `ledger/runs_history.csv` | id / one-liner / commit / cv / status | append one row per run (ok or fail) |
| `ledger/CURRENT_STRATEGY.md` | executable spec | **rewrite in full** each plan |
| `ledger/runs/<id>.json` | metrics-only JSON | write after each run; never planner-visible |
| `logs/<id>.log` | long traces | disk only; never chat or RESULTS |

`LEARNINGS.md` and `STRATEGY.md` are planner-always memory (compact kill reasons + queue).
Do not paste fold arrays or traces into them.

## Python — loop harness (`src/loop/`)

- Small functions, obvious names, few layers of abstraction.
- Adapters are thin shell / HTTP wrappers. No TUI automation.
- Stdlib + PyYAML only in the loop runtime.
- Prefer `--dry-run` mocks over hitting real CLIs in CI.

## Python — competition library (`src/ev_s6e9/`)

- Prefer **human-readable** code over clever code.
- Prefer **small modular functions** and as few LOC as possible while staying clear.
- **Feature engineering** and **modeling** each get their **own classes**
  (`FeatureBuilder`, model factories). Train / predict / `scripts/run_exp.py` call those APIs.
- Type hints on public functions. Docstrings only when they add something.
- No god-files. Do not move logic into notebooks.
- Independent seeds / variants / folds: process-parallelize. Keep `n_jobs=-1`
  (do not cap per-worker threads). Do not switch `tree_method` / `device`
  or seeds / params for speed. Do not rewrite `src/loop/` for wall-clock.

## Rewind-safe `src/ev_s6e9/` (forbidden pattern)

Each iteration restores `exps/` + `src/ev_s6e9/` to the CSV-best CV commit
**before** plan/execute. `src/loop/`, prompts, config, and ledgers stay put.

- Prefer new work via `exps/expNNNN/config.json` + existing CLI flags
  (`scripts/run_exp.py` / `python -m ev_s6e9 train …`).
- **Do not** rely on leftover edits in `src/ev_s6e9/` from a prior iteration
  (`--orig`, `load_original()`, extra columns, etc.). They will not be there.
- If a strategy truly needs library code changes: implement them **in the same
  execute turn before train**, keep them minimal/modular, and leave them on
  allow-listed paths so the run commit (ok or fail) records them.
- Prefer experiment-only helpers under `exps/<id>/` or thin config-callable
  wrappers rather than growing core APIs for one-off ideas — when practical.
- Never invent columns. `Years_of_Driving_Experience` is not in `schema.py`.
  Check `TRAIN_COLS` / the train header only.

## Experiments (Deotte / BirdCLEF factory)

- CV is ground truth. Record LB only after a real submit.
- One change per iteration: copy last **keep** (`exps/exp0010/` at seed) → `exps/expNNNN/` → edit the copy.
- Train with `python scripts/run_exp.py expNNNN` (uses `config.json` → `python -m ev_s6e9 train`).
- Submit when CV beats the CSV-best (any status with a valid CV):
  `python scripts/submit_if_improved.py expNNNN` (or `python -m loop submit-if-improved`).
  `--dry-run` prints keep/kill without Kaggle or git. `--gate lb` compares public LB
  after submit. Never stages `oof.csv`, data CSVs, joblib, `.env`, or secrets.
  The overnight loop calls this after every scored run, including ``status=fail``.
- Keep `seed` / 5-fold stratified split unless STRATEGY says otherwise. Metric = ROC AUC.
- Write OOF / models under `outputs/` or the exp folder. **Do not commit** `oof.csv`, `*.joblib`, or Kaggle CSVs.
- To regenerate exp0010 OOF: `python scripts/run_exp.py exp0010` after `python -m ev_s6e9 download`.
- Phase order unless RESULTS show a reason to jump: EDA → baseline → FE → stack.

## CLI agents (long loop — not Cursor)

- Planner primary: `agy` headless (`--model` **before** `-p`). See `agy models`.
- Planner fallback: OpenAI-compatible DeepSeek (`DEEPSEEK_BASE_URL`).
- Executor: `qwen -p` with `OPENAI_BASE_URL` / `OPENAI_MODEL` pointing at local Qwen ~27B.
- Do not drive the overnight loop through Cursor. The old Fable `autoloop.py` path is **not** in this repo.
