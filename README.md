# EV S6E9 Agent Loop

Filled **competition instance** of the token-thrifty
[kaggle-agent-loop](https://github.com/wTaylorBickelmann/kaggle-agent-loop)
for [playground-series-s6e9](https://github.com/wTaylorBickelmann/ev-purchase-kaggle)
(*Predicting Electric Vehicle Purchases*, ROC AUC, target `Will_Buy_EV`).

| Repo | Role |
|------|------|
| [kaggle-agent-loop](https://github.com/wTaylorBickelmann/kaggle-agent-loop) | Reusable **empty shell**: Antigravity planner → DeepSeek fallback → Qwen executor, whitelist, ledgers. Do not vendor competition code there. |
| **This repo** | S6E9 **instance**: that loop plus the vendored `src/ev_s6e9/` library, seeded at **exp0010** keep **CV 0.94552 / LB 0.94561** (from `ev-purchase-kaggle` @ `a4c75ae`). |

The overnight loop runs **locally in this checkout**. There is no recurring pull from
`ev-purchase-kaggle`. The old Cursor-Fable `autoloop.py` path is not used.

## How the pieces fit

One iteration is **planner → ledgers → executor → train → RESULTS**, then the next
plan reads those ledgers. The planner sees only `config/planner_reads.yaml`.

```
+---------------------------+
| config/planner_reads.yaml |
| (whitelist only; no logs) |
+-------------+-------------+
              |
              v
+-------------+-------------+       recoverable fail
| PLANNER                   |---------------------+
| Antigravity (`agy`)       |                     |
+-------------+-------------+                     v
              |                       +-----------+-----------+
              | ok                    | FALLBACK              |
              |                       | DeepSeek (local HTTP) |
              |                       +-----------+-----------+
              |                                   |
              +----------------+------------------+
                               |
                               v
              +----------------+------------------+
              | LEDGERS                           |
              | rewrite  CURRENT_STRATEGY.md      |
              | append   STRATEGIES.md            |
              +----------------+------------------+
                               |
                               v
              +----------------+------------------+
              | EXECUTOR                          |
              | Qwen Code (`qwen -p`, local ~27B) |
              +----------------+------------------+
                               |
                               v
              +----------------+------------------+
              | TRAIN                             |
              | copy keep  exps/exp0010           |
              |         -> exps/expNNNN           |
              | scripts/run_exp.py                |
              | python -m ev_s6e9 train           |
              +----------------+------------------+
                               |
              +----------------+------------------+
              | RESULTS                           |
              | append   ledger/RESULTS.md        |
              | write    ledger/runs/<id>.json    |
              | write    logs/<id>.log (disk only)|
              +----------------+------------------+
                               |
                               v
                         next iteration
```

`--dry-run` swaps mock planner + executor (no `agy` / `qwen` / GPU). Same ledger
writes; CV is the fake `0.5`. A shorter copy of this picture lives in
`docs/architecture.txt` and `CURSOR.md`.

## Floor (exp0010)

Chris Deotte Fable 5.1 3×XGBoost blend (`--strategy deotte`) plus:

- `--freq` — value counts for `Annual_Income_USD` and `Daily_Commute_km`
- `--te` — fold-safe smoothed target encoding (m=20, nested OOF) of exact `Annual_Income_USD`

Config / NOTES / metrics live in `exps/exp0010/`. **OOF CSVs are not committed**
(~18 MB). Regenerate after downloading data:

```bash
python -m ev_s6e9 download
python scripts/run_exp.py exp0010
# writes outputs/oof.csv and copies metrics into exps/exp0010/
```

`ledger/CURRENT_STRATEGY.md` is the **next** one-change after that keep
(multi-seed 42/43/44 on the exp0010 recipe). The planner rewrites it each iteration.

## Install

```bash
git clone https://github.com/wTaylorBickelmann/ev-s6e9-agent-loop
cd ev-s6e9-agent-loop
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Smoke-test (no CLIs, no GPU, no network, no data):

```bash
python -m loop run --iterations 1 --dry-run
python -m loop show-whitelist
```

## Mac Studio kickoff

1. Python 3.11+, [Antigravity CLI](https://antigravity.google/docs/cli/headless/) `agy`,
   [Qwen Code](https://qwenlm.github.io/qwen-code-docs/) `qwen`, and Ollama (or vLLM).
2. Pull a DeepSeek planner model and a Qwen **~27B** executor model:

   ```bash
   ollama pull deepseek-r1:32b    # or your local DeepSeek tag
   ollama pull qwen3.6:27b        # or qwen2.5-coder:32b / …
   ```

3. Authenticate `agy` once (`agy models` should list a Gemini Flash-class slug).
   Pin it in `.env` (`AGY_MODEL=gemini-3.5-flash-high`, `AGY_EFFORT=high`).
4. Point Qwen at Ollama:

   ```bash
   OPENAI_BASE_URL=http://127.0.0.1:11434/v1
   OPENAI_MODEL=qwen3.6:27b
   OPENAI_API_KEY=ollama
   ```

5. Download competition CSVs (gitignored) with the existing CLI:

   ```bash
   # ~/.kaggle/kaggle.json (chmod 600) + join playground-series-s6e9
   python -m ev_s6e9 download
   ```

6. Confirm the planner whitelist, then run:

   ```bash
   python -m loop show-whitelist
   python -m loop run --iterations 20
   # or
   ./scripts/run-loop.sh 20
   ```

`competition.root` is **this repo** (`.` / `COMPETITION_ROOT=.`). The executor copies
`exps/exp0010/` → `exps/expNNNN/`, trains via `scripts/run_exp.py`, and appends one
`ledger/RESULTS.md` row.

Suggested memory split: do **not** keep both 32B DeepSeek and 27B Qwen resident if
unified memory is tight. They run in sequence (plan, then execute).

Useful commands:

```bash
python -m loop plan-once              # rewrite CURRENT_STRATEGY.md only
python -m loop execute-once           # run the current spec
python -m loop run --iterations 5
python -m loop run --iterations 1 --dry-run
python -m ev_s6e9 train --strategy deotte --freq --te
python scripts/submit_if_improved.py exp0019 --dry-run   # gate on CV; no submit/commit
python -m loop submit-if-improved exp0019               # submit + commit only on CV best
```

Submit only on a **CV personal best** (repo convention: CV is ground truth). The helper
reads `exps/<exp>/metrics.json`, compares to the best keep (keep `metrics.json`, RESULTS
`ok` rows, or the exp0010 floor **0.94552**), and on a lift of more than `--eps` (default
`1e-5`) runs predict if needed, `python -m ev_s6e9 submit`, polls public LB with a timeout,
appends LB into RESULTS / EXPERIMENTS, then `git add`s only allow-listed paths (exp
config/NOTES/metrics, ledgers, `src/` — never data CSVs, `oof.csv`, joblib, `.env`).
`--gate lb` compares public LB after submit instead. `--push` is off by default.
A kill prints `kill` and exits `1` with no submit and no commit.

Stop conditions live in `config/loop.yaml`: `max_iterations`, `target_cv`,
`max_consecutive_failures`.

## Token-budget rules

Copied from the shell — prefer these over the old Fable/Cursor planner path:

1. **Whitelist is law.** Planner context = `config/planner_reads.yaml` only.
2. **Never pass `logs/`**, `data/`, `oof.csv`, or notebook outputs to the planner.
3. **Ledgers stay tabular.** One strategy line, one result row.
4. **Byte caps.** Default 24 KiB/file and 80 KiB total.
5. **No Cursor for the long loop.** `python -m loop run` on the Mac Studio.

Always-inlined: ledgers, `CURSOR.md`, `LEARNINGS.md`, `STRATEGY.md`.
Optional: `src/ev_s6e9/{features,deotte,model,train}.py`, exp0010 NOTES/config,
`STRATEGIES.md`, Deotte/TOP20 notes. Denied: `logs/`, `data/`, `outputs/`, OOF dumps.

## Layout

Flow: **planner → ledgers → executor → exps/train → RESULTS** (diagram above).

```
src/loop/            Antigravity / DeepSeek / Qwen adapters + orchestrator
src/ev_s6e9/         vendored S6E9 library (features, deotte, train, predict, …)
exps/exp0010/        keep floor (config / NOTES / metrics; regenerate OOF locally)
scripts/run_exp.py             train one exp folder
scripts/submit_if_improved.py  submit + commit only on a CV (or LB) personal best
scripts/run-loop.sh            python -m loop run
ledger/              compact STRATEGIES / RESULTS / CURRENT_STRATEGY
config/loop.yaml     competition.root = .
docs/architecture.txt
```

## Tests / CI

```bash
ruff check src tests
pytest
python -m loop run --iterations 1 --dry-run
```

CI runs lint, pytest (loop unit tests + EV library smokes on synth data), and the
dry-run. No `agy`, `qwen`, GPU, or Kaggle CSVs.

`--dry-run` from the repo root appends a mock `s0NN` row to the sample ledgers.
Use `--root /tmp/some-copy` if you want to keep starters clean.
