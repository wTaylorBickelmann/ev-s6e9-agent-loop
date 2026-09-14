# STRATEGY — human-owned experiment queue

Standing plan for the autonomous loop (playground-series-s6e9, ROC-AUC).

**Files are memory.** The planner is inlined only the whitelist (`config/planner_reads.yaml`):
ledgers, this file, `LEARNINGS.md`, `CURSOR.md`, and a few `src/ev_s6e9/` + exp0010 NOTES.
Do not expect `logs/` or `reports/kills/` in context.

## North star
- Beat local best CV, then climb toward Deotte public ~**0.94672**.
- Do **not** optimize the public LB in the inner loop. Submit only on CV personal best (orchestrator).

## Forbidden (leak / self-cheat)
- Do not change the **metric** (must stay ROC-AUC on fixed stratified 5-fold unless STRATEGY says otherwise).
- Do not change the **fold split** (seed 42, n=5 stratified on target) without an explicit STRATEGY line.
- Do not peek at test labels / use test target.
- Do not rewrite old `reports/index.md` rows or old `EXPERIMENTS.md` chunks.
- Do not “fix” a bad score by tightening early stopping only to inflate OOF without a real idea.
- Do not plan/execute a strategy that depends on leftover `src/ev_s6e9/` edits from a **previous** iteration. Rewind restores that tree to CSV-best every turn. Prefer `exps/expNNNN/config.json` + existing CLI flags. If a library change is required, it must land in the **same** execute turn as train (and be commitable). Do not invent columns (`Years_of_Driving_Experience` is not in `schema.py`).

## One-change rule
Each iteration: **exactly one** hypothesis. Copy last accepted exp → new folder → edit the copy only (plus minimal shared lib **in this same execute turn** if a new strategy flag is required — rewind will wipe uncommitted `src/ev_s6e9/` leftovers before the next plan).

## Queue (human edits this; agent consumes top unchecked)

Priority order — try in order, skip items already in `reports/index.md` as keep or kill:

1. [ ] Multi-seed Deotte blend (seeds 42/43/44), equal average OOF/test
2. [ ] Deotte blend weight search on OOF (m1/m2/m3 weights ≠ 1/3)
3. [ ] Add CatBoost or LightGBM as 4th model in blend with Deotte XGBs
4. [ ] Stronger recipe features from original-data EDA (interactions already partially in)
5. [ ] Target-encode high-cardinality cats with fold-safe TE
6. [ ] Hill-climb stack: LGBM OOF + Deotte OOF logistic/ridge meta on OOF only
7. [ ] Mine `TOP20_PUBLIC_NOTEBOOKS.md` for one concrete unused idea

## Stop lines (orchestrator / human)
- `STOP=1` in `outputs/loop_state.json` or max iters / no-improve streak.
- Daily Kaggle submit budget exhausted.
- Human sets **pause** — do not invent stop reasons to quit early.

## Agent role vs human role
| Human | Agent |
|------|--------|
| This file, CV protocol, leakage | Implement one change in new `exps/expNNNN/` |
| When to stop stacking junk | Train via orchestrator, log metrics |
| Which public notebook is real baseline | Append LEARNINGS on failure |

## Already tried (do not re-queue as-is)

| Idea | Where | Verdict |
|------|--------|---------|
| Identical Deotte clone | exp0001 | kill |
| 3-seed on **pre-freq/TE** Deotte | exp0001 | kill vs 0.94210; **not** tried on exp0010 |
| Extra freq/age/nmode | exp0002 | kill vs 0.94333 |
| LGBM as 4th blend model | exp0008 | kill vs 0.94333 |
| OOF m1/m2/m3 weight search | exp0009 | kill vs 0.94333 |
| Income TE (m=20) + freq | **exp0010** | **keep** CV 0.94552 / LB 0.94561 |
| + commute TE / te-m=2 / lr=0.02 / pair TE / orig TE | exp0011–16 | kill vs exp0010 |

Queue item 1 (multi-seed) is the next one-change **on the exp0010 recipe**. Items 2–3
were killed on the weaker 0.94333 floor; do not revive them until 3-seed on exp0010
is scored. Item 5 (TE) is done for income; further TE variants are killed.

## Current accepted floor
**exp0010** `deotte --freq --te` (income TE m=20): CV **0.94552 ± 0.00064**, public LB **0.94561**.
See `exps/exp0010/`, `ledger/RESULTS.md`, `reports/index.md`.
