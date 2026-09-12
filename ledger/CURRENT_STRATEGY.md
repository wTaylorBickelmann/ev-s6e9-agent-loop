# s023 — 3-seed Deotte blend on exp0010 floor

## Hypothesis
Averaging predictions across three distinct random seeds (42, 43, 44) reduces tree variance on the exp0010 floor (--freq --te), lifting CV AUC over 0.94552.

## Changes
- files to touch: src/ev_s6e9/features.py, src/ev_s6e9/deotte.py, exps/exp0023/config.json
- features / model / CV protocol: Revert FeatureBuilder.recipe_score and recipe_logit in features.py to baseline Deotte (remove home_chg, restore offset 5.5); run Deotte 3×XGB across seeds 42, 43, 44; average seed OOF predictions evaluated on fixed 5-fold StratifiedKFold (seed 42)
- runtime budget: ~25-30 minutes on CPU (3× 5-fold Deotte CV runs)

## Acceptance
- metric: roc_auc
- beat or inform: best CV 0.94552 (exp0010); keep if CV > 0.94552, else kill

## Logging
- write long output only to logs/s023.log
- RESULTS.md gets one table row; never paste logs
