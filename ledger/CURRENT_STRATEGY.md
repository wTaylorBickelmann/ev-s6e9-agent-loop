# s044 — Multi-seed averaged TE encoding

## Hypothesis
Current TE OOF values use a single inner KFold seed (42). For medium-count income/commute values (count 10–50), the inner fold assignment materially changes the TE estimate; this injects encoding noise that propagates identically across all 10 model seeds. Averaging TE OOF across 3 inner seeds (42, 43, 44) reduces this variance. cstdy (LB 0.94647, rank 17) lists multiple TE seeds as part of their pipeline.

## Changes
- files: `src/ev_s6e9/features.py`, `src/ev_s6e9/deotte.py`
- features.py: add `te_oof_avg(vals, y, *, seeds=(42,43,44), folds=5, m=20.0)` → calls `te_oof` per seed, returns `np.mean(results, axis=0)`
- deotte.py `_fit_fold`: replace all 6 `te_oof(...)` calls (income m=20/5/2, commute m=20/5/2) with `te_oof_avg(...)` using same `m` param; val/test `te_apply` unchanged (maps fit on full outer fold, no inner seed)
- ~15 lines touched; no new deps
- train_args: `--strategy deotte --freq --te --seeds 42,43,44,45,46,47,48,49,50,51 --weight-search`
- CV protocol: 5-fold seed 42, 10 model seeds (42-51), 3 variants × weight search — identical to s041
- runtime: TE overhead 3× (~seconds per fold); XGB fits unchanged; total ≈ 20 min

## Acceptance
- metric: roc_auc
- beat: s041 CV 0.94575
- kill if CV ≤ 0.94575

## Logging
- write long output only to logs/s044.log
- RESULTS.md gets one table row; never paste logs
