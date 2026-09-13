# Results

One row per executed run. Metrics only. Full traces live at `logs/<id>.log`.
Seeded compactly from reports/index.md + EXPERIMENTS.md @ a4c75ae. Floor = s010 / exp0010.

| id | status | cv | lb | notes |
|----|--------|----|----|-------|
| s000 | ok | 0.94210 | 0.94182 | exp0000 Deotte 3×XGB equal blend; early floor; 5-fold seed 42 |
| s001 | ok | 0.94333 | — | exp0001 +freq income/commute; kept then superseded by exp0010 |
| s008 | fail | 0.94336 | — | exp0008 +LGBM m4; +0.00003 vs 0.94333 — kill (noise) |
| s009 | fail | 0.94333 | — | exp0009 blend-weight search; no lift vs equal 1/3 |
| s010 | ok | 0.94552 | 0.94561 | exp0010 deotte --freq --te income; KEEP floor; parent=exp0001 |
| s011 | fail | 0.94557 | — | exp0011 TE commute + income; kill vs exp0010 (no real lift) |
| s012 | fail | 0.94559 | — | exp0012 te-m=2; kill vs exp0010 |
| s013 | fail | 0.94555 | — | exp0013 lr=0.02; kill vs exp0010 |
| s015 | fail | 0.94551 | — | exp0015 income×commute pair TE; kill vs exp0010 |
| s016 | fail | 0.94553 | — | exp0016 orig_income_te; kill vs exp0010 |
| s018 | fail | — | — | exp0018 planner-fail (old Cursor Fable path); not a model result |
| s019 | fail | — | — | qwen exit 1: [API Error: Context is too large to send safely after automatic compression. Estimated prompt tokens: 34698; hard limit: 30852.8; compression status: COMPRESSION_FAILED_EMPTY_SUMMARY. Start a new sess |
| s020 | fail | — | — | qwen exit -15: Operation cancelled. |
| s021 | fail | — | — | qwen exit 1: [API Error: Context is too large to send safely after automatic compression. Estimated prompt tokens: 30940; hard limit: 30852.8; compression status: COMPRESSION_FAILED_EMPTY_SUMMARY. Start a new sess |
| s022 | fail | 0.94549 | — | exp0022 recipe +1.5 home-charge, offset 7.0; 0.94549 < floor 0.94552 (kill) |
| s023 | ok | 0.94556 | 0.94556 | exp0023 3-seed blend 42/43/44; CV 0.94556 ± 0.00062 vs floor 0.94552 (KEEP; manual train after qwen loop-guard) |
| s024 | ok | 0.94558 | — | exp0024 weight search m1/m2/m3 (w=0.276/0.621/0.103); CV 0.94558 ± 0.00062 vs 0.94556 (KEEP) |
| s025 | fail | 0.94557 | — | exp0025 +income_x_concern helper; CV 0.94557 ± 0.00063 < 0.94558 (kill) |
| s026 | ok | 0.94529 | — | exp0026 LightGBM baseline on Deotte freq+TE; CV 0.94529 ± 0.00059 (inform: standalone LGBM capacity, below floor 0.94552) |
| s027 | fail | — | — | qwen exit 124: timeout |
| s028 | fail | 0.94551 | — | exp0028 +commute_x_anxiety helper; CV 0.94551 ± 0.00064 < 0.94558 floor (kill) |
| s029 | ok | 0.94536 | — | exp0029 CatBoost 5-fold baseline on Deotte freq+TE; CV 0.94536 ± 0.00064 (inform: standalone CatBoost capacity, below floor 0.94558) |
| s030 | fail | 0.94558 | — | exp0030 ensemble CatBoost(exp0029)+Deotte XGB(exp0024) alpha=0.14; CV 0.94558 ± 0.00062 = floor (kill: no lift) |
| s031 | ok | 0.94559 | — | exp0031 no_home_charge_x_high_anxiety helper (replaces commute_x_anxiety); CV 0.94559 ± 0.00063 vs floor 0.94558 (marginal +0.00001, within noise) |
| s032 | ok | 0.94308 | — | exp0032 HGB baseline (max_iter=50, lr=0.05, 31 leaves) on Deotte freq+TE; CV 0.94308 ± 0.00066 (inform: standalone sklearn HGB capacity, below floor 0.94559) |
| s033 | fail | 0.94557 | — | exp0033 +commute_per_charger helper (Daily_Commute_km/(home+work+1)); CV 0.94557 ± 0.00062 < 0.94559 floor (kill) |
| s035 | fail | 0.94557 | — | exp0035 +income_trailing_zeros +income_last_digit; CV 0.94557 ± 0.00060 < 0.94559 floor (kill) |
| s036 | ok | 0.94561 | — | exp0036 5-seed (42-46) Deotte blend on clean s031 5-helper feature set + weight search; CV 0.94561 ± 0.00063 vs floor 0.94559 (marginal +0.00002, within noise) |
| s037 | ok | 0.94566 | — | exp0037 multi-smoothing TE income (m=5 + m=20) 5-seed blend + weight search; CV 0.94566 ± 0.00064 vs floor 0.94561 (KEEP +0.00005) |
| s038 | ok | 0.94567 | — | exp0038 10-seed (42-51) Deotte blend on dual-TE (m=5+m=20) + weight search; CV 0.94567 ± 0.00063 vs floor 0.94566 (marginal +0.00001, within noise) |
| s039 | ok | 0.94567 | — | exp0039 3-model OOF rank-blend (XGB exp0038 + LGBM exp0026 + CatBoost exp0029) w=0.897/0.034/0.069; CV 0.94567 ± 0.00063 = floor (kill: weight search collapsed onto XGB, no lift over prob-blend control 0.94567) |
| s040 | ok | 0.94569 | — | exp0040 triple-smoothing TE income (m=2+m=5+m=20) 10-seed blend + weight search; CV 0.94569 ± 0.00062 vs floor 0.94567 (marginal +0.00002, within noise) |
| s041 | ok | 0.94575 | 0.88273 | recovered from exps/exp0041/metrics.json (executor timed out after training succeeded) |
| s042 | ok | 0.94543 | — | exp0042 LGBM on full s041 6-TE features (income m=20/5/2 + commute m=20/5/2); CV 0.94543 ± 0.00064; +0.00014 vs s026 single-TE; gap to s041 XGB = 0.00032 (below 0.94555 stack threshold) |
| s043 | fail | 0.94566 | — | exp0043 XGB max_depth=8 on s041 feature set; CV 0.94566 ± 0.00064 < floor 0.94575 (kill: depth 6 already optimal for this feature count) |
| s044 | fail | — | — | qwen exit 1: Loop detection halted the run (consecutive_identical_tool_calls: the model repeated the same tool call with identical arguments). This is an always-on guard and cannot be disabled via `model.skipLoopD |
