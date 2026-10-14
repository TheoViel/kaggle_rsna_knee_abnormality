"""Menisci / per-finding check of a run against a baseline run, alone and inside the current recipe, run from src/.

    PYTHONPATH=. python ../src_bis/oai_joint_eval.py --run 2026-10-13/50 --base 2026-10-11/30 [--findings ...]

Per finding, 58 / holdout:
  run, base                      each alone
  recipe                         ext.finding_eval.RECIPE (validation counterparts of the package)
  recipe, base -> run            the recipe with the baseline replaced by the run (if the base is in the recipe)
  recipe + run                   the run added to the recipe
"""
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

import ext.finding_eval as fe
from knee_cls.data import load_tables, make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--findings", default="Medial Meniscus,Lateral Meniscus")
a = ap.parse_args()
st, _ = load_tables()
_, gt, hold = make_splits(st, 0.1, 0)
st = st.set_index("StudyInstanceUID")


def sc(p, c):
    g, h = p.reindex(gt), p.reindex(hold)
    return f"{roc_auc_score(st.loc[gt, 'gt_' + c], g.fillna(g.median())):.3f} / " \
           f"{roc_auc_score(st.loc[hold, 'target_' + c] >= .5, h.fillna(h.median())):.3f}"


rows = []
for c in a.findings.split(","):
    rec = fe.RECIPE.get(c, [])
    run, base = fe.study(a.run, c), fe.study(a.base, c)
    row = {"finding": c, "run": sc(run, c), "base": sc(base, c)}
    if rec:
        row["recipe"] = sc(fe.rank_mean([fe.study(r, c) for r in rec]), c)
        if a.base in rec:
            row["recipe, base -> run"] = sc(fe.rank_mean([fe.study(a.run if r == a.base else r, c) for r in rec]), c)
        row["recipe + run"] = sc(fe.rank_mean([fe.study(r, c) for r in rec] + [run]), c)
    rows.append(row)
print(f"run {a.run} vs base {a.base} (58 / holdout)")
print(pd.DataFrame(rows).set_index("finding").fillna("-").to_string())
