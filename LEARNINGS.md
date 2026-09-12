# LEARNINGS — do not repeat

Compact kill list for the planner. One short bullet per failed or wasted idea.
Do not paste logs or local paths.

Template:
```
- YYYY-MM-DD expNNNN: <idea> → CV <score> vs best <best> — <why kill>
```

---

- (seed) 2026-09-10: bare hyperparam jitter without structural change is low-EV on this AUC plateau
- 2026-09-10 exp0001: deotte-baseline (identical train_args) → CV 0.94210 vs 0.94210 — kill
- 2026-09-11 exp0001: deotte-3seed-blend on **old** Deotte baseline (no freq/TE) → 0.94212 vs 0.94210 — kill *that parent only*; retrying 3-seed **on exp0010** is still the queue head
- 2026-09-11 exp0002: deotte-freq2-age-nmode → 0.94328 vs 0.94333 — kill
- 2026-09-12 exp0008: deotte-plus-lgbm-m4 → 0.94336 vs 0.94333 — kill (noise; 4th-model idea not proven)
- 2026-09-12 exp0009: deotte-blend-weight-search → 0.94333 vs 0.94333 — kill (equal 1/3 is enough on pre-TE blend)
- 2026-09-12 exp0011: deotte-te-commute → 0.94557 vs **0.94552 exp0010** — kill
- 2026-09-12 exp0012: deotte-te-m2 → 0.94559 vs 0.94552 — kill
- 2026-09-12 exp0013: deotte-lr02 → 0.94555 vs 0.94552 — kill
- 2026-09-12 exp0015: deotte-te-pair → 0.94551 vs 0.94552 — kill
- 2026-09-12 exp0016: deotte-orig-te → 0.94553 vs 0.94552 — kill
- 2026-09-12 exp0014/17/18: Cursor-Fable planner timeouts/exits — do not use that planner path; this instance uses Antigravity→Qwen
- Executor timeouts (3600s) on early TE/m4 attempts: keep wall-clock in CURRENT_STRATEGY; write traces only to `logs/`
