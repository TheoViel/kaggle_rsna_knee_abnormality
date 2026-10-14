"""Inference-time series selection per finding, on the dataset_8 recipe's validation counterparts; run from src/.

    python -m ext.plane_eval [--out ../output/locations/plane_eval.csv]

For every finding, each validation model's per-series predictions (val_series_preds.csv) are filtered to a subset
of series, averaged per study (per-ROI sub-targets "<ROI>|fracture": series mean per ROI, then max over ROIs), and
the models rank-averaged as in the recipe. Subsets: all (as in the package), no axial, axial only, sagittal only,
coronal only, fluid-sensitive (FS) only, FS without axial, non-FS only. A study with no series in the subset keeps
its all-series score (fallback). Scores: AUC on the 58 radiologist studies and on the holdout (Claude labels >= 0.5;
not meaningful for synovitis, whose models learn the OAI image convention), and + b1_c1 (50/50 rank, 58 only) for
the findings the notebook blends.
"""

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from crops.train import finding_of
from knee_cls.data import load_tables, make_splits

R = "../logs/crops_run/"
o, x, d, e, t = "2026-10-09/", "2026-10-10/", "2026-10-11/", "2026-10-12/", "2026-10-13/"
RECIPE = {"ACL": [d + "65", o + "75"], "MCL": [d + "91", o + "75"],
          "Medial Meniscus": [d + "30", x + "93", o + "75"], "Lateral Meniscus": [d + "30", x + "93@Sagittal,Coronal", o + "75"],
          "Medial OA": [o + "75", d + "33", d + "17"], "Lateral OA": [d + "33"], "PF OA": [d + "17"], "Baker's": [o + "84", o + "75"],
          "Fracture": [e + "25", e + "26"], "Contusion": [e + "58", t + "1", t + "9"],
          "Synovitis": [t + "44", t + "49", t + "77", t + "79"], "Effusion": [x + "27", t + "87"]}
BLENDED = ("Fracture", "Contusion", "Synovitis", "Effusion")
SUBSETS = {"all": lambda s: s.plane.notna(), "no axial": lambda s: s.plane != "Axial", "axial only": lambda s: s.plane == "Axial",
           "sagittal only": lambda s: s.plane == "Sagittal", "coronal only": lambda s: s.plane == "Coronal",
           "FS only": lambda s: s.fs == 1, "FS, no axial": lambda s: (s.fs == 1) & (s.plane != "Axial"),
           "non-FS only": lambda s: s.fs == 0}


def model_scores(ref, finding, fs):
    run, _, planes = ref.partition("@")
    v = pd.read_csv(R + run + "/val_series_preds.csv")
    sub = v.cls.str.contains("|", regex=False)
    v = v[(v.cls == finding) | (sub & (v.cls.str.split("|").str[1].fillna("").map(finding_of) == finding))].copy()
    if planes:
        v = v[v.plane.isin(planes.split(","))]
    v["fs"] = v.SeriesInstanceUID.map(fs)
    return v


def study(v, keep):
    def agg(q):
        p = q.groupby(["StudyInstanceUID", "cls"]).p.mean().reset_index()
        return p.groupby("StudyInstanceUID").p.max()          # one class, or max over the per-ROI sub-targets
    full = agg(v)
    return agg(v[keep(v)]).reindex(full.index).fillna(full)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../output/locations/plane_eval.csv")
    a = ap.parse_args()
    st, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(st, holdout_frac=0.1, seed=0)
    S = st.set_index("StudyInstanceUID")
    fs = pd.read_csv("../input/train_series.csv").set_index("SeriesInstanceUID").Fluid_Sensitive
    b1 = pd.read_csv("../input/b1_c1_sealed_58_predictions.csv").set_index("StudyInstanceUID")
    rows = []
    for c, refs in RECIPE.items():
        G = S.loc[gt_ids, f"gt_{c}"].dropna().round()
        H = (S.loc[hold_ids, f"target_{c}"].dropna() >= 0.5).astype(int)
        V = [model_scores(r, c, fs) for r in refs]
        for name, keep in SUBSETS.items():
            p = sum(study(v, keep).rank(pct=True) for v in V)
            r = dict(finding=c, subset=name, gt=roc_auc_score(G, p.reindex(G.index)),
                     hold=roc_auc_score(H, p.reindex(H.index)),
                     share_fallback_58=float(1 - pd.concat([v[keep(v)] for v in V]).StudyInstanceUID.isin(G.index).groupby(
                         pd.concat([v[keep(v)] for v in V]).StudyInstanceUID).any().reindex(G.index).fillna(False).mean()))
            if c in BLENDED:
                q = (p.reindex(G.index).rank(pct=True) + b1[c].reindex(G.index).rank(pct=True)) / 2
                r["gt + b1_c1"] = roc_auc_score(G, q)
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    for c in RECIPE:
        print(f"\n== {c}")
        print(d[d.finding == c].drop(columns="finding").set_index("subset").round(3).to_string())


if __name__ == "__main__":
    main()
