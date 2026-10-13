"""Series rules x model sets on the 58 / holdout (offline, validation runs' per-series predictions)."""
from evaluate import Scorer, combine, report, SUBMITTED, VAL, study_scores, series_preds
import pandas as pd, numpy as np, itertools
S = Scorer()
ROI8 = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Baker's"]
EXT = {"Effusion": ["jc1"], "Synovitis": ["jc1"], "Contusion": ["jc1"], "Fracture": ["fr9a"]}
sets = {
    "submitted + jc1/fr9a": {**SUBMITTED, **EXT},
    "submitted + jc1,jc2/fr9a,b": {**SUBMITTED, **{c: ["jc1", "jc2"] for c in EXT}, "Fracture": ["fr9a", "fr9b"]},
    "88 only + jc1/fr9a": {**{c: ["88"] for c in ROI8}, **EXT},
    "88 only + j27/fr9a": {**{c: ["88"] for c in ROI8}, **{c: ["j27"] for c in EXT}, "Fracture": ["fr9a"]},
}
for rule in ["all", "all_sc", "fs", "nofs", "most", "least", "first"]:
    keep = S.pick(rule)
    print(f"== {rule}: {len(keep)} series")
    for n, r in sets.items():
        res, m58, mh = S.auc(combine(r, keep, S.index))
        report("  " + n, res, m58, mh)
