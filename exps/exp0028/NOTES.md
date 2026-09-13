# exp0028 — deotte-commute-x-anxiety

**Parent:** exp0010 (Deotte `--freq --te`, CV 0.94552 ± 0.00064)

**Hypothesis:** Range anxiety in EV adoption is acutely operationalized by daily driving
distance. Adding a `commute_x_anxiety` interaction helper to the Deotte `FeatureBuilder`
informs the trees that High range anxiety suppresses purchase probability primarily when
`Daily_Commute_km` is large, lifting CV beyond the 0.94558 floor (s024).

**Change:** one new helper column `commute_x_anxiety = Daily_Commute_km * [Range_Anxiety_Level == "High"]`
added to `HELPER_COLS` and `FeatureBuilder._helper_frame` in `src/ev_s6e9/features.py`.
Everything else (XGB params, 5-fold seed 42, recipe, base_margin, 1/3 blend, `--freq --te`)
unchanged. This is a single-feature "one change" on the exp0010 base recipe.

**Acceptance:** beat or inform best CV 0.94558 (s024). CV is the score that matters.

**Run:**

```bash
python scripts/run_exp.py exp0028
```

Long output goes to `~/.cache/ev-s6e9-agent-loop/logs/s028.log`. Do not commit the OOF CSV.
