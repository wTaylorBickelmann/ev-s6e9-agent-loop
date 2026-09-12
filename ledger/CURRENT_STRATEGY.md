# s019 — multi-seed Deotte blend on exp0010 floor

Playground Series S6E9 (Predicting Electric Vehicle Purchases). Metric: ROC AUC.
Last **keep** is `exps/exp0010/` (`deotte --freq --te`, CV **0.94552**, public LB **0.94561**).

## Hypothesis

A previous 3-seed blend (`exp0001` / `deotte-3seed-blend`, seeds 42/43/44) was killed at
**0.94212 vs 0.94210** on the *old* Deotte baseline (no freq, no income TE). That kill does
not apply to the exp0010 floor. Averaging three independent seed runs of the **current**
recipe (freq + fold-safe income TE, same 3×XGB blend) should reduce seed variance and may
lift OOF AUC by ≥ 0.00010 vs 0.94552. This is the highest-priority unchecked `STRATEGY.md`
queue item that has not been tried on the current keep.

Skip (already killed on this floor or older, do not retry as this change): extra TE
columns (commute / pair / orig), te-m=2, lr=0.02, LGBM-as-m4, OOF weight search on the
pre-TE blend. Those stay in LEARNINGS.

## Changes

- Copy `exps/exp0010/` → `exps/exp0019/` (next free exp id after exp0018 planner-fail).
- Parent: exp0010. One change only: **multi-seed average**, not new features or a 4th model.
- Keep `--strategy deotte --freq --te`, folds=5, outer split seed **42**, XGB defaults
  (`n_estimators=3000`, `lr=0.05`, `max_depth=6`).
- Train the same Deotte 3-model blend at seeds `{42, 43, 44}` (or equivalent `train_args`
  plus a small library/script hook if a `--seeds` flag is required).
- Equal-average the three seed OOF vectors and the three test prediction vectors.
- Do not change the fold protocol, metric, recipe, TE column, or blend (still 1/3 m1/m2/m3
  *inside* each seed). Do not add CatBoost/LGBM. Do not hill-climb.
- Entry points: `src/ev_s6e9/deotte.py`, `src/ev_s6e9/__main__.py`, `scripts/run_exp.py`.
- Runtime: three full Deotte CVs — expect tens of minutes on CPU, not hours. Write long
  output only to `logs/s019.log`.

`exps/exp0019/config.json` should look like exp0010 plus a `seeds` list
`[42, 43, 44]`, `parent` exp0010, title `deotte-te-income-3seed`, same
`train_args` (`--strategy deotte --freq --te`), folds 5.

Train via `python scripts/run_exp.py exp0019` after `python -m ev_s6e9 download`
(if `data/raw/train.csv` is missing). Do not commit `oof.csv` or fold models.

## Acceptance

- Metric: `roc_auc` (higher is better).
- Produce mean OOF AUC ± std (and per-seed means if cheap).
- **Keep** if CV ≥ 0.94562 (about +0.00010 vs exp0010 0.94552); else **kill**.
- Append one RESULTS row for `s019`; write `ledger/runs/s019.json` (metrics only).
- Do not submit unless this becomes a new personal-best keep.

## Logging

- Never paste fold arrays, OOF, or traces into the ledger.
- Kill reason (if any) goes as one LEARNINGS bullet.
