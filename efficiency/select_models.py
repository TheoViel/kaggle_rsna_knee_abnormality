"""Per-finding model / plane choice under the 1 sagittal + 1 coronal rule, with a GPU cost model.

cost (ms of H100 fp16 per study) = sum over entries of n_planes x n_crops x ms_per_crop (bench.py);
runtime on 2 x T4 ~ cost x 1e-3 x 650 studies per GPU x 10 (T4 / H100).
Objective per finding (efficiency units): -AUC_f / 12 / 0.4594 + cost_s / 32400.
"""
import itertools, sys
import numpy as np, pandas as pd
from evaluate import Scorer, combine
MS = {"88": .262, "91": .262, "mr85": .262, "pl86": .262, "mk90": .262, "fr9a": .262 * 9, "fr9b": .262 * 9,
      "cafax87": .425, "mk87": 1.087, "mk89": .777, "jc1": 1.18, "j27": 1.18}
CAND = {"ACL": ["mr85", "88"], "MCL": ["pl86", "88"], "Medial Meniscus": ["mk87", "cafax87", "88"],
        "Lateral Meniscus": ["mk87", "cafax87", "88"], "Medial OA": ["88", "mk89", "mk90"],
        "Lateral OA": ["mk89", "mk90", "88"], "PF OA": ["mk90", "mk89", "88"], "Baker's": ["88", "91"],
        "Effusion": ["jc1", "j27"], "Synovitis": ["jc1", "j27"], "Contusion": ["jc1", "j27"], "Fracture": ["fr9a", "fr9b", "jc1"]}
PL = {"": 2, "@Sagittal": 1, "@Coronal": 1}
S = Scorer(); keep = S.pick("fs")
T4 = 1e-3 * 650 * 10
crit = sys.argv[1] if len(sys.argv) > 1 else "mean"
rows = []
for f, cands in CAND.items():
    entries = [m + p for m in cands for p in PL]
    for k in (1, 2, 3):
        for combo in itertools.combinations(entries, k):
            if len({e.split("@")[0] for e in combo}) < k:
                continue
            p = combine({f: list(combo)}, keep, S.index)
            res, a58, ah = S.auc(p)
            cost = sum(PL["@" + e.split("@")[1] if "@" in e else ""] * MS[e.split("@")[0]] for e in combo)
            if f in ("Effusion", "Synovitis", "Contusion") or (f == "Fracture" and all(e.startswith(("jc1", "j27")) for e in combo)):
                cost /= 4            # joint crop: one encoder pass shared by its 4 findings
            rows.append(dict(finding=f, combo="+".join(combo), a58=a58, ah=ah, ms=cost, sec=cost * T4))
R = pd.DataFrame(rows)
R["auc"] = {"mean": (R.a58 + R.ah) / 2, "58": R.a58, "hold": R.ah}[crit]
R["eff"] = -R.auc / 12 / 0.4594 + R.sec / 32400
R.to_csv("select_models.csv", index=False)
pd.set_option("display.width", 200)
for f, g in R.groupby("finding", sort=False):
    g = g.sort_values("eff")
    print(f"== {f}")
    print(g.head(6)[["combo", "a58", "ah", "ms", "sec", "eff"]].round(4).to_string(index=False))
