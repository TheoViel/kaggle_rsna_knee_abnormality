"""Per-finding check of a run against b1_c1 and the current package's recipe, with blends; run from src/.

    python -m ext.finding_eval --run 2026-10-13/28 [--run2 2026-10-13/40]

For every finding the run predicts (its val_study_preds.csv; --run2 averages a second seed in, by rank), prints
on the 58 radiologist studies (and the holdout, Claude labels >= 0.5):
  run              the run alone
  b1_c1            b1_c1 (input/b1_c1_sealed_58_predictions.csv, 58 only)
  recipe           the current package's recipe for that finding (dataset_2_moaks_mrnet_men_frac_cont, its
                   validation counterparts; - where the package has no model and b1_c1 is used)
  run + b1_c1      50/50 rank average (58 only)
  run + recipe     the run added to the recipe's models (rank average)
"""

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables, make_splits

R = "../logs/crops_run/"
o, d, e, x = "2026-10-09/", "2026-10-11/", "2026-10-12/", "2026-10-10/"
RECIPE = {"ACL": [d + "65", o + "75"], "MCL": [d + "91", o + "75"],
          "Medial Meniscus": [d + "30", x + "93", o + "75"], "Lateral Meniscus": [d + "30", x + "93@Sagittal,Coronal", o + "75"],
          "Medial OA": [o + "75", d + "33", d + "17"], "Lateral OA": [d + "33"], "PF OA": [d + "17"], "Baker's": [o + "84", o + "75"],
          "Fracture": [e + "25", e + "26"], "Contusion": [e + "58", "2026-10-13/1", "2026-10-13/9"]}


def study(ref, c):
    run, _, planes = ref.partition("@")
    if planes:
        v = pd.read_csv(R + run + "/val_series_preds.csv")
        v = v[(v.cls == c) & v.plane.isin(planes.split(","))]
        return v.groupby("StudyInstanceUID").p.mean()
    t = pd.read_csv(R + run + "/val_study_preds.csv")
    return t.set_index(t.columns[0])[c]


def rank_mean(series):
    return sum(s.rank(pct=True) for s in series) / len(series)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--run2", default="")
    a = ap.parse_args()
    st, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(st, holdout_frac=0.1, seed=0)
    S = st.set_index("StudyInstanceUID")
    b1 = pd.read_csv("../input/b1_c1_sealed_58_predictions.csv").set_index("StudyInstanceUID")
    t = pd.read_csv(R + a.run + "/val_study_preds.csv")
    findings = [c for c in t.columns[1:] if f"gt_{c}" in S]
    rows = []
    for c in findings:
        runs = [a.run] + ([a.run2] if a.run2 else [])
        p = rank_mean([study(r, c) for r in runs])
        G = S.loc[gt_ids, f"gt_{c}"].dropna().round()
        H = (S.loc[hold_ids, f"target_{c}"].dropna() >= 0.5).astype(int)

        def sc(q, hold=True):
            g = roc_auc_score(G, q.reindex(G.index))
            return f"{g:.3f}" + (f" / {roc_auc_score(H, q.reindex(H.index)):.3f}" if hold and q.reindex(H.index).notna().all() else "")
        bc = b1[c].rank(pct=True) if c in b1 else None
        rec = rank_mean([study(r, c) for r in RECIPE[c]]) if c in RECIPE else None
        rows.append({"finding": c, "run": sc(p),
                     "b1_c1": sc(bc, False) if bc is not None else "-",
                     "recipe": sc(rec) if rec is not None else "-",
                     "run + b1_c1": sc((p.reindex(G.index).rank(pct=True) + bc.reindex(G.index)) / 2, False) if bc is not None else "-",
                     "run + recipe": sc(rank_mean([study(r, c) for r in RECIPE[c] + runs])) if rec is not None else "-"})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("finding").to_string())


if __name__ == "__main__":
    main()
