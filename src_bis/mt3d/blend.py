"""Score mt3d runs alone, against the main agent's references, and blended into them, run from src_bis/.

    python -m mt3d.blend --runs 2026-10-10/0,2026-10-10/1

References, rebuilt from the non-fullfit runs' validation predictions
(logs/crops_run/<run>/val_study_preds.csv), per finding rank averages:
  - `all`    the single 8-ROI 2.5D model 09/75;
  - `recipe` the dataset_5 recipe for the 8 ROI findings (summary.md), lateral OA
             minus 0.4 x medial OA ranks, then re-ranked.
mt3d runs (logs/bis/mt3d/<run>) are rank-averaged together (`mt3d`), then blended
into each reference at weight w: (1 - w) * rank(ref) + w * rank(mt3d).

AUC on the 58 radiologist studies (`gt`) and on the holdout against Claude labels
binarised at 0.5 (`hold`), as in crops.train.study_auc.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables
from roi.extract import ROI_CLASSES

CROPS = "../logs/crops_run/"
MT = "../logs/bis/mt3d/"
RECIPE = {"ACL": ["10/65", "10/77", "10/78", "10/79"], "MCL": ["09/80", "09/75"],
          "Medial Meniscus": ["09/81", "09/75"], "Lateral Meniscus": ["09/81", "09/75"],
          "Medial OA": ["10/37", "10/40", "10/42", "10/43"], "Lateral OA": ["10/37", "10/40", "10/42", "10/43"],
          "PF OA": ["09/83"], "Baker's": ["09/75", "09/84"]}


def rank(s):
    return s.rank(pct=True)


def load(path):
    return pd.read_csv(os.path.join(path, "val_study_preds.csv"), index_col=0)


def rank_avg(tabs, c, idx):
    r = [rank(t[c].reindex(idx)) for t in tabs if c in t]
    return pd.concat(r, axis=1).mean(1)


def lateral_fix(p):
    p = dict(p)
    p["Lateral OA"] = rank(p["Lateral OA"] - 0.4 * p["Medial OA"])
    return p


def score(p, st, idx):
    gt_m = st.loc[idx, [f"gt_{c}" for c in ROI_CLASSES]].notna().all(1)
    out = {}
    for c in ROI_CLASSES:
        x = p[c].fillna(p[c].mean())
        g = st.loc[idx, f"gt_{c}"]
        h = (st.loc[idx, f"target_{c}"] >= 0.5).astype(int)
        out[c] = dict(gt=roc_auc_score(g[gt_m], x[gt_m]), hold=roc_auc_score(h[~gt_m], x[~gt_m]))
    return pd.DataFrame(out).T


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--runs", required=True)
    a.add_argument("--weights", default="0.25,0.5")
    args = a.parse_args()
    studies, _ = load_tables()
    st = studies.set_index("StudyInstanceUID")
    mt = [load(MT + r) for r in args.runs.split(",")]
    idx = mt[0].index
    cache = {}

    def run(r):
        if r not in cache:                                   # "09/75" -> logs/crops_run/2026-10-09/75
            cache[r] = load(f"{CROPS}2026-10-{r}")
        return cache[r]

    ref = {"all": {c: rank(run("09/75")[c].reindex(idx)) for c in ROI_CLASSES},
           "recipe": lateral_fix({c: rank_avg([run(r) for r in RECIPE[c]], c, idx) for c in ROI_CLASSES})}
    m = {c: rank_avg(mt, c, idx) for c in ROI_CLASSES}
    rows = {"mt3d": score(m, st, idx), "mt3d + LOA fix": score(lateral_fix(m), st, idx)}
    for name, p in ref.items():
        rows[name] = score(p, st, idx)
        for w in map(float, args.weights.split(",")):
            b = {c: (1 - w) * p[c] + w * m[c] for c in ROI_CLASSES}
            if name == "recipe":
                b["Lateral OA"] = (1 - w) * rank_avg([run(r) for r in RECIPE["Lateral OA"]], "Lateral OA", idx) \
                    + w * m["Lateral OA"]
                b = lateral_fix(b)
            rows[f"{name} + {w:g} mt3d"] = score(b, st, idx)
    pd.set_option("display.width", 250)
    for k in ("gt", "hold"):
        t = pd.DataFrame({n: s[k] for n, s in rows.items()}).T
        t.columns = [c[:9] for c in t.columns]
        t["MEAN"] = t.mean(1)
        print(f"\n== {k}\n{t.round(3).to_string()}")


if __name__ == "__main__":
    main()
