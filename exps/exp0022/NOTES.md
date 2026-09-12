# exp0022 — deotte-recipe-homecharge

**Parent:** exp0010 (CV 0.94552 ± 0.00064)

**Hypothesis (s022):** `Home_Charging_Possible` is a strong generative driver of EV
purchase. Adding it to `recipe_score` (+1.5) and raising the `recipe_logit` offset to
7.0 aligns m2 (base_margin) and m3 (recipe feature) with the true data-generating
process, lifting blend CV over exp0010's 0.94552.

**Change:** `src/ev_s6e9/features.py`
- `FeatureBuilder.recipe_score`: `+ 1.5 * _yes(Home_Charging_Possible)`
- `FeatureBuilder.recipe_logit`: offset `5.5` -> `7.0`

Net effect on base_margin: home-charging rows keep their prior (score +1.5, offset +1.5
cancel); non-home-charging rows get a lower prior. XGB params, folds (5, seed 42),
`--freq --te`, and the 1/3 blend are unchanged.

**Acceptance:** keep if CV >= 0.94562, else kill (floor = exp0010 0.94552).

```bash
python scripts/run_exp.py exp0022
```

That rewrites `outputs/oof.csv`, `outputs/cv.json`, and copies metrics back here.
Do not add the OOF CSV to git.
