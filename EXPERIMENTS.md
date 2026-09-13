# Experiments

Append-only running log for playground-series-s6e9. **Newest entries at the bottom.**

Append a chunk after every attempt (or let `python -m ev_s6e9 train` / submit helpers do it). Keep it terse. Do not edit old entries.

**Rule:** once Kaggle returns a public score for that attempt, the chunk must include it under `LB:` (poll `kaggle competitions submissions` after submit if needed). Leave `LB: —` only while scoring is still pending.

Template:

```
### YYYY-MM-DD — <model / features / key params>
- CV: <mean AUC ± std>
- LB: <public score or — while pending>
- Takeaway: <one line>
```

---

### 2026-09-09 — LightGBM baseline (pipeline seed)
- Model: LightGBM, 5-fold stratified, raw cols + `charging_total`
- Params: n_estimators=800, lr=0.05, num_leaves=31, early_stopping=50
- CV: pending — run `python -m ev_s6e9 train` after `download` (appends a scored chunk below)
- LB: —
- Takeaway: first baseline scaffold; fill CV from a real-data train run
### 2026-09-09 — LightGBM n_estimators=800 lr=0.05 num_leaves=31, raw+charging_total, 5-fold
- CV: 0.94171 ± 0.00073
- LB: —
- Takeaway: auto-logged from train

### 2026-09-09 — first Kaggle submit (same LGBM baseline)
- CV: 0.94171 ± 0.00073
- LB: 0.94150
- Takeaway: CV≈LB; solid first score, room vs ~0.95 leaders

### 2026-09-09 — Chris Deotte Fable 5.1 strategy ported (`--strategy deotte`)
- Model: 3× XGB (baseline + recipe base_margin + recipe feature), equal-weight blend; helpers + EV recipe from original-data EDA
- Params: n_estimators=3000 lr=0.05 max_depth=6, 5-fold seed 42 (see STRATEGIES.md)
- CV: — (library + synth smoke only; run full `train --strategy deotte` on real data)
- LB: —
- Takeaway: Deotte path listed in STRATEGIES.md; full-data score still needed
### 2026-09-10 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94210 ± 0.00075
- LB: —
- Takeaway: Chris Deotte Fable 5.1 full-data train

### 2026-09-10 — Kaggle submit deotte xgb 3-model blend (Fable 5.1)
- CV: 0.94210 ± 0.00075
- LB: 0.94182
- Takeaway: CV≈LB; tiny lift vs LGBM baseline (0.94150), still well below Deotte public 0.94672

### 2026-09-11 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94333 ± 0.00070
- LB: —
- Takeaway: exp0001: +freq-encoded income/commute (train+test value counts) in Deotte blend

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94336 ± 0.00070
- LB: —
- Takeaway: exp0008: +LightGBM m4 (num_leaves=63) as 4th equal-weight model in Deotte blend

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94333 ± 0.00070
- LB: —
- Takeaway: exp0009: OOF-searched m1/m2/m3 blend weights (nested LOFO grid, step 0.05) on top of exp0001 --freq

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94552 ± 0.00064
- LB: —
- Takeaway: exp0010: +fold-safe smoothed TE (m=20, nested OOF) of exact Annual_Income_USD on top of --freq

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94557 ± 0.00061
- LB: —
- Takeaway: exp0011: +fold-safe TE (m=20, nested OOF) of exact Daily_Commute_km alongside income TE, on top of --freq

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94559 ± 0.00065
- LB: —
- Takeaway: exp0012: income TE smoothing m=20 -> 2 (new --te-m flag), on top of --freq

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.02 max_depth=6), 5-fold
- CV: 0.94555 ± 0.00064
- LB: —
- Takeaway: exp0013: XGB learning_rate 0.05 -> 0.02 (new --lr flag), on top of --freq --te income

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94551 ± 0.00063
- LB: —
- Takeaway: exp0015: +fold-safe smoothed TE (m=20, nested OOF) of (Annual_Income_USD, Daily_Commute_km) pair key on top of --freq --te

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94553 ± 0.00063
- LB: —
- Takeaway: exp0016: +orig_income_te (mean original-dataset label per exact Annual_Income_USD) on top of --freq --te

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94549 ± 0.00063
- LB: —
- Takeaway: exp0022: recipe_score +1.5*Home_Charging_Possible, recipe_logit offset 7.0 (on top of --freq --te)

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94556 ± 0.00062
- LB: —
- Takeaway: exp0023: 3-seed Deotte blend (42,43,44) on exp0010 floor, reverted recipe

### 2026-09-12 — exp0023 submit
- CV: 0.94556
- LB: 0.94556
- Takeaway: exp0023 submitted CV=0.94556 LB=0.94556

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94558 ± 0.00062
- LB: —
- Takeaway: exp0024: Deotte m1/m2/m3 weight search on 3-seed OOF

### 2026-09-12 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94557 ± 0.00063
- LB: —
- Takeaway: exp0025: +income_x_concern helper feature (affluent-green interaction)

### 2026-09-12 — LightGBM n_estimators=800 lr=0.05 num_leaves=31, Deotte freq+TE, 5-fold
- CV: 0.94529 ± 0.00059
- LB: —
- Takeaway: exp0026: LightGBM baseline on Deotte freq+TE features (5-fold seed 42)

### 2026-09-13 — Ensemble LGBM+Deotte (alpha=0.110), 5-fold
- CV: 0.94558 ± 0.00061
- LB: —
- Takeaway: exp0027: ensemble exp0026 LGBM + exp0024 Deotte 3-seed XGB via OOF weight search

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94551 ± 0.00064
- LB: —
- Takeaway: exp0028: +commute_x_anxiety helper (commute * High range anxiety)

### 2026-09-13 — LightGBM n_estimators=1500 lr=0.05 num_leaves=None, Deotte freq+TE, 5-fold
- CV: 0.94536 ± 0.00064
- LB: —
- Takeaway: exp0029: CatBoost 5-fold baseline on Deotte freq+TE features

### 2026-09-13 — Ensemble LGBM+Deotte (alpha=0.140), 5-fold
- CV: 0.94558 ± 0.00062
- LB: —
- Takeaway: exp0030: ensemble exp0029 CatBoost + exp0024 Deotte 3-seed XGB via OOF weight search

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94559 ± 0.00063
- LB: —
- Takeaway: exp0031: no_home_charge_x_high_anxiety helper (replaces commute_x_anxiety)

### 2026-09-13 — LightGBM n_estimators=None lr=0.05 num_leaves=None, raw+charging_total, 5-fold
- CV: 0.94308 ± 0.00066
- LB: —
- Takeaway: exp0032: HGB baseline on Deotte freq+TE features

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94557 ± 0.00062
- LB: —
- Takeaway: exp0033: +commute_per_charger helper (Daily_Commute_km / (home+work+1))

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94557 ± 0.00060
- LB: —
- Takeaway: exp0035: +income_trailing_zeros +income_last_digit helper features

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94561 ± 0.00063
- LB: —
- Takeaway: exp0036: 5-seed (42-46) Deotte blend on clean s031 5-helper feature set + weight search

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94566 ± 0.00064
- LB: —
- Takeaway: exp0037: multi-smoothing TE income (m=5 + m=20) 5-seed blend + weight search

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94567 ± 0.00063
- LB: —
- Takeaway: exp0038: 10-seed (42-51) Deotte blend on dual-TE (m=5+m=20) + weight search

### 2026-09-13 — 3-model OOF rank-blend (w=0.897/0.034/0.069), 5-fold
- CV: 0.94567 ± 0.00063
- LB: —
- Takeaway: exp0039: 3-model OOF rank-blend (XGB exp0038 + LGBM exp0026 + CatBoost exp0029) with simplex weight search (n_grid=30)

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94569 ± 0.00062
- LB: —
- Takeaway: s040 triple-TE m=2+m=5+m=20

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94575 ± 0.00064
- LB: —
- Takeaway: s041 +commute triple-TE m=2+m=5+m=20

### 2026-09-13 — LightGBM n_estimators=800 lr=0.05 num_leaves=31, Deotte freq+TE, 5-fold
- CV: 0.94543 ± 0.00064
- LB: —
- Takeaway: s042 LGBM on full s041 6-TE features

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=8), 5-fold
- CV: 0.94566 ± 0.00064
- LB: —
- Takeaway: s043 XGB max_depth=8 on s041 feature set

### 2026-09-13 — exp0041 submit
- CV: 0.94575
- LB: 0.88273
- Takeaway: exp0041 submitted CV=0.94575 LB=0.88273

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94567 ± 0.00061
- LB: —
- Takeaway: s044 multi-seed averaged TE (3 inner seeds 42/43/44) on s041 feature set

### 2026-09-13 — Deotte XGB 3-model blend (n_estimators=3000 lr=0.05 max_depth=6), 5-fold
- CV: 0.94567 ± 0.00061
- LB: —
- Takeaway: s041 RETRAIN triple-TE commute+income m2+m5+m20

### 2026-09-13 — exp0041 submit
- CV: 0.94575
- LB: 0.94580
- Takeaway: exp0041 submitted CV=0.94575 LB=0.94580

