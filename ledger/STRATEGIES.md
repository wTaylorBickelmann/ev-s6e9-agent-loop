# Strategies

Append-only catalog. One row per planned idea. Never paste logs.
Seeded from ev-purchase-kaggle @ a4c75ae (exp0010 keep). Next id is after the last table row.

| id | date | phase | one-liner |
|----|------|-------|-----------|
| s000 | 2026-09-10 | baseline | exp0000 Deotte Fable 5.1 3×XGB equal blend (early floor CV 0.94210) |
| s001 | 2026-09-11 | fe | exp0001 +freq-encoded income/commute on Deotte blend (kept, then superseded) |
| s008 | 2026-09-12 | stack | exp0008 +LightGBM m4 as 4th equal-weight model (killed vs 0.94333) |
| s009 | 2026-09-12 | stack | exp0009 OOF blend-weight search m1/m2/m3 (killed, no lift) |
| s010 | 2026-09-12 | fe | exp0010 fold-safe income TE (m=20) on --freq Deotte — KEEP floor |
| s011 | 2026-09-12 | fe | exp0011 +TE Daily_Commute_km beside income TE (killed vs exp0010) |
| s012 | 2026-09-12 | fe | exp0012 income TE smoothing m=20→2 (killed vs exp0010) |
| s013 | 2026-09-12 | fe | exp0013 XGB lr 0.05→0.02 on --freq --te (killed vs exp0010) |
| s015 | 2026-09-12 | fe | exp0015 TE of (income, commute) pair key (killed vs exp0010) |
| s016 | 2026-09-12 | fe | exp0016 orig-data income target mean as feature (killed vs exp0010) |
| s018 | 2026-09-12 | fe | exp0018 Cursor-Fable planner fail (log only; do not retry that path) |
| s019 | 2026-09-12 | stack | 3-seed Deotte blend (seeds 42, 43, 44) on exp0010 floor |
| s020 | 2026-09-12 | stack | 3-seed Deotte blend (seeds 42, 43, 44) on exp0010 floor |
| s021 | 2026-09-12 | eda | EDA recipe update: add Home_Charging_Possible (+1.5) to recipe_score and base_margin on exp0010 floor |
| s022 | 2026-09-12 | eda | EDA recipe update: add Home_Charging_Possible (+1.5) to recipe_score and base_margin on exp0010 floor |
| s023 | 2026-09-12 | stack | 3-seed Deotte blend (seeds 42, 43, 44) on exp0010 floor with reverted baseline recipe |
| s024 | 2026-09-12 | stack | Find optimal OOF blend weights for m1/m2/m3 across seeds 42/43/44 vs equal 1/3 on exp0023 |
| s025 | 2026-09-12 | eda | Add income_x_concern interaction helper feature from original-data EDA to Deotte FeatureBuilder |
| s026 | 2026-09-12 | baseline | LightGBM 5-fold baseline on Deotte features (--freq --te) to benchmark standalone non-XGB performance |
| s027 | 2026-09-12 | stack | Ensemble exp0026 LightGBM OOF with exp0024 3-seed Deotte XGB blend via OOF grid search |
| s028 | 2026-09-12 | eda | Add commute_x_anxiety helper feature from EDA to Deotte FeatureBuilder on exp0010 floor |
| s029 | 2026-09-12 | baseline | CatBoost 5-fold baseline on Deotte features (--freq --te) to benchmark standalone CatBoost performance |
| s030 | 2026-09-12 | stack | Ensemble exp0029 CatBoost OOF with exp0024 3-seed Deotte XGB blend via OOF grid search |
| s031 | 2026-09-12 | eda | Add no_home_charge_x_high_anxiety interaction helper from EDA to Deotte FeatureBuilder on exp0024 floor |
| s032 | 2026-09-12 | baseline | HistGradientBoosting 5-fold baseline on Deotte features (--freq --te) to benchmark standalone sklearn GBDT capacity |
| s033 | 2026-09-12 | fe | Add commute_per_charger ratio feature (Daily_Commute_km / (chargers_total + 1)) to FeatureBuilder on exp0031 floor |
| s034 | 2026-09-13 | eda | Add subsidy_x_high_anxiety interaction helper to Deotte FeatureBuilder on exp0031 floor |
| s035 | 2026-09-13 | fe | Add income digit features (trailing zeros, last digit) exploiting synthetic data artifacts on s031 floor |
| s036 | 2026-09-13 | stack | 5-seed Deotte blend (42-46) on cleaned s031 features (drop 3 killed helpers) + weight search |
| s037 | 2026-09-13 | fe | Add second income TE (m=5) alongside m=20 for multi-smoothing on s036 5-seed blend |
| s038 | 2026-09-13 | stack | 10-seed Deotte blend (42-51) on s037 dual-TE features + weight search |
| s039 | 2026-09-13 | stack | Rank-blend 10-seed XGB (s038) + LGBM (s026) + CatBoost (s029) OOFs with 3-way weight search |
| s040 | 2026-09-13 | fe | Add income TE m=2 (3rd smoothing) to 10-seed Deotte blend + weight search |
| s041 | 2026-09-13 | fe | Triple-smoothing TE on Daily_Commute_km (m=2+m=5+m=20) beside income triple-TE, 10-seed + weight search |
| s042 | 2026-09-13 | baseline | Retrain LightGBM 5-fold on s041 full feature set (6 TEs + helpers + freq) to benchmark updated LGBM capacity |
| s043 | 2026-09-13 | baseline | XGB max_depth 8 (from 6) on s041 6-TE features, 10-seed Deotte blend + weight search |
| s044 | 2026-09-13 | fe | Multi-seed averaged TE (inner seeds 42-44) for all 6 TE columns, reducing OOF encoding noise |
