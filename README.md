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

One iteration is **rewind → plan → train → RESULTS+CSV → commit → maybe submit**,
then the next plan reads those ledgers. The planner sees only `config/planner_reads.yaml`.

```
+-----------------------------------------------+
| REWIND                                        |
| read ledger/runs_history.csv                  |
| pick row with best CV (any status; skip null) |
| git checkout <that SHA> -- exps/ src/ev_s6e9/ |
| src/loop, prompts, config, ledger stay put    |
+----------------------+------------------------+
                       |
                       v
+----------------------+------------------------+
| config/planner_reads.yaml (whitelist; no logs)|
+----------------------+------------------------+
                       |
                       v
+----------------------+--------+   recoverable fail
| PLANNER                       |------------------+
| Antigravity (`agy`)           |                  |
+----------------------+--------+                  v
                       | ok              +---------+-----------+
                       |                 | FALLBACK            |
                       |                 | DeepSeek (local HTTP)|
                       |                 +---------+-----------+
                       |                           |
                       +-------------+-------------+
                                     |
                                     v
                       +-------------+-------------+
                       | LEDGERS                   |
                       | rewrite  CURRENT_STRATEGY |
                       | append   STRATEGIES.md    |
                       +-------------+-------------+
                                     |
                                     v
                       +-------------+-------------+
                       | EXECUTOR                  |
                       | Qwen Code (`qwen -p`)     |
                       +-------------+-------------+
                                     |
                                     v
                       +-------------+-------------+
                       | TRAIN                     |
                       | copy CSV-best -> expNNNN  |
                       | scripts/run_exp.py        |
                       | python -m ev_s6e9 train   |
                       +-------------+-------------+
                                     |
                       +-------------+-------------+
                       | RECORD                    |
                       | append RESULTS.md         |
                       | append runs_history.csv   |
                       | write  ledger/runs/<id>   |
                       +-------------+-------------+
                                     |
                       +-------------+-------------+
                       | ALWAYS COMMIT             |
                       | loop: RUN <id> status= cv=|
                       | (ok or fail; SHA -> CSV)  |
                       +-------------+-------------+
                                     |
                          CV beats CSV-best + eps?
                          (fail + valid CV counts)
                          /                    \
                        yes                     no
                         |                       |
                         v                       v
              +----------+-----------+      next iteration
              | SUBMIT-IF-IMPROVED   |
              | predict + Kaggle     |
              | loop: KEEP (+submit) |
              +----------+-----------+
                         |
                         v
                   next iteration
```

`--dry-run` swaps mock planner + executor (no `agy` / `qwen` / GPU). Same ledger
+ CSV writes; CV is the fake `0.5`; no git rewind/commit/submit. A shorter copy
lives in `docs/architecture.txt` and `CURSOR.md`.

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
2. Serve DeepSeek with the **llama-server agent harness** and have a Qwen **~27B** executor ready:

   ```bash
   # Planner fallback — V4-Flash Q3 GGUF + built-in tools (--agent)
   bash scripts/serve_deepseek_harness.sh
   # Executor model (Ollama)
   ollama pull qwen3.8:27b-q4_K_M   # or your local Qwen ~27B tag
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

`competition.root` is **this repo** (`.` / `COMPETITION_ROOT=.`). Each iteration
rewinds `exps/`+`src/ev_s6e9/` to the CSV-best CV commit (not `src/loop/`), the executor trains via
`scripts/run_exp.py`, then the loop appends RESULTS + `runs_history.csv` and commits.

Suggested memory split: do **not** keep both DeepSeek (~120GB Q3) and 27B Qwen resident
at once. They run in sequence (plan, then execute). Bounce backends **without** restarting
the loop process:

```bash
./scripts/restart_agent.sh status
./scripts/restart_agent.sh swap-to-executor    # DeepSeek down, Qwen up
./scripts/restart_agent.sh swap-to-planner     # Qwen down; DeepSeek stays cold (agy primary)
./scripts/restart_agent.sh deepseek up|down|restart
./scripts/restart_agent.sh qwen up|down|restart
./scripts/restart_agent.sh executor bounce     # kill stuck qwen CLI only; loop stays up
```

`executor bounce` does **not** re-run the same strategy — the in-flight `execute_once`
fails closed and the orchestrator continues. Retry manually with
`python -m loop execute-once` if needed. DeepSeek is fallback-only; leave it down while
`agy` is healthy.

Useful commands:

```bash
python -m loop plan-once              # rewrite CURRENT_STRATEGY.md only
python -m loop execute-once           # run the current spec
python -m loop run --iterations 5
python -m loop run --iterations 1 --dry-run
python -m ev_s6e9 train --strategy deotte --freq --te
python scripts/submit_if_improved.py exp0019 --dry-run   # gate on CSV-best CV; no submit/commit
python -m loop submit-if-improved exp0019               # submit + KEEP commit on CSV-best
```

### Real train + submit (local Mac Studio)

Needs competition CSVs (`python -m ev_s6e9 download`) and Kaggle credentials
(`~/.kaggle/kaggle.json` or `~/.kaggle/access_token`). The cloud CI VM does **not**
have this data — use `--dry-run` there.

```bash
# Train one exp folder (writes metrics / OOF under exps/ and outputs/)
# Optional workers: EV_S6E9_TRAIN_WORKERS or --max-workers (default min(jobs, CPUs)).
# Scores are unchanged (same seeds/params → same OOF/AUC); only wall-clock drops.
python scripts/run_exp.py exp0041

# Submit only if this CV beats ledger/runs_history.csv (score-first, fail OK)
python -m loop submit-if-improved exp0041 --dry-run   # print keep/kill
python -m loop submit-if-improved exp0041             # predict + Kaggle + KEEP commit

# Overnight loop: rewind → plan → train → CSV+commit → submit-if-best
python -m loop run --iterations 20
```

Submit when the run is a **CV personal best** (repo convention: CV is ground truth).
The helper reads `exps/<exp>/metrics.json`, compares to the **CSV-best CV** in
`ledger/runs_history.csv` (any status with a valid score — a timeout-after-success
`fail` still counts), falling back to scored exp/RESULTS rows or the exp0010 floor
**0.94552**. On a lift of more than `--eps` (default `1e-5`, `loop.submit_eps` in
`config/loop.yaml`) it runs predict if needed, `python -m ev_s6e9 submit`, polls
public LB with a timeout, appends LB into RESULTS / EXPERIMENTS, then `git add`s
only allow-listed paths (exp config/NOTES/metrics, ledgers including the CSV,
`src/` — never data CSVs, `oof.csv`, joblib, `.env`). The overnight loop calls
this after every scored run, **including `status=fail`**. `--gate lb` compares
public LB after submit instead. `--push` is off by default. A kill prints `kill`
and exits `1` with no submit and no extra KEEP commit (the RUN commit already happened).

Stop conditions live in `config/loop.yaml`: `max_iterations`, `target_cv`,
`max_consecutive_failures`.

## Token-budget rules

Copied from the shell — prefer these over the old Fable/Cursor planner path:

1. **Whitelist is law.** Planner context = `config/planner_reads.yaml` only.
2. **Never pass `logs/`**, `data/`, `oof.csv`, or notebook outputs to the planner.
3. **Ledgers stay tabular.** One strategy line, one result row.
4. **Byte caps.** Default 24 KiB/file and 80 KiB total.
5. **No Cursor for the long loop.** `python -m loop run` on the Mac Studio.
6. **Traces live outside the git tree.** `LOOP_LOGS_DIR` (default
   `~/.cache/ev-s6e9-agent-loop/logs`) so Qwen Code cannot bulk-ingest run logs from cwd.
   In-repo `logs/` is only a pointer (`logs/README.md`). Add `.qwenignore` patterns for
   `data/`, `*.csv`, `oof`, etc.

Always-inlined: ledgers, `CURSOR.md`, `LEARNINGS.md`, `STRATEGY.md`.
Optional: `src/ev_s6e9/{features,deotte,model,train}.py`, exp0010 NOTES/config,
`STRATEGIES.md`, Deotte/TOP20 notes. Denied: `logs/`, `data/`, `outputs/`, OOF dumps.

## Layout

Flow: **rewind-from-CSV-best → plan → execute/train → RESULTS+CSV → always commit →
maybe submit if best** (diagram above).

```
src/loop/            Antigravity / DeepSeek / Qwen adapters + orchestrator
src/ev_s6e9/         vendored S6E9 library (features, deotte, train, predict, …)
exps/exp0010/        keep floor (config / NOTES / metrics; regenerate OOF locally)
scripts/run_exp.py             train one exp folder
scripts/submit_if_improved.py  submit + KEEP commit on a CSV-best CV (or LB)
scripts/run-loop.sh            python -m loop run
ledger/              STRATEGIES / RESULTS / CURRENT_STRATEGY / runs_history.csv
config/loop.yaml     competition.root = . ; loop.submit_eps
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
