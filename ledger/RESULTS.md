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
