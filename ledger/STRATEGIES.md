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
