# exp0024 — Deotte m1/m2/m3 weight search

- parent: exp0023
- seeds: 42,43,44 with --freq --te
- weight search: grid (n=30) over simplex (w1,w2,w3 >= 0, sum=1)
- learned weights: m1=0.2759, m2=0.6207, m3=0.1034
- CV: 0.94558 ± 0.00062
- floor exp0023: 0.94556
- delta: +0.00002
- decision: KEEP (CV > 0.94556)
- note: m2 (base_margin) dominates; m3 (recipe_feature) gets minimal weight
