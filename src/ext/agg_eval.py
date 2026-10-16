"""Series -> study aggregation, on the dataset_10 recipe's validation counterparts; run from src/.

    python -m ext.agg_eval [--out ../output/locations/agg_eval.csv]

The package scores a study as the mean over its series of each model's per-series probability (per-ROI fracture
targets: mean over series per ROI, then max over ROIs), then rank-averages the recipe's models. This replaces the
mean over series by other aggregations of the same per-series predictions (val_series_preds.csv) and scores every
finding on the 58 radiologist studies and the holdout (Claude labels >= 0.5; not meaningful for synovitis, whose
models follow the image convention). Lateral OA is the lc group without the medial correction.

Aggregations of a study's series probabilities p_1..p_n:
  mean          the package (arithmetic mean)
  logit mean    sigmoid(mean(logit p))
  median
  max           the most confident series
  top-2 mean    the mean of the 2 highest
  power-2/4     (mean p^k)^(1/k): between mean and max
  plane mean    the mean per plane, then the mean over planes (each plane counts once, whatever its series count)
  plane max     the mean per plane, then the max over planes
  noisy-OR      1 - prod(1 - p)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crops.train import finding_of
from knee_cls.data import load_tables, make_splits

R = "../logs/crops_run/"
o, x, d, e, t, f = "2026-10-09/", "2026-10-10/", "2026-10-11/", "2026-10-12/", "2026-10-13/", "2026-10-14/"
RECIPE = {"ACL": [d + "65", o + "75"], "MCL": [d + "91", o + "75"],
          "Medial Meniscus": [d + "30", x + "93", o + "75"], "Lateral Meniscus": [d + "30", x + "93@Sagittal,Coronal", o + "75"],
          "Medial OA": [o + "75", d + "33", d + "17"], "Lateral OA": [f + "70", f + "72", f + "73"], "PF OA": [d + "17"],
          "Baker's": [o + "84", o + "75"], "Fracture": [t + "94", t + "98"], "Contusion": [e + "58", t + "1", t + "9"],
          "Synovitis": [t + "44", t + "49", t + "77", t + "79"], "Effusion": [x + "27", t + "87"]}
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


AGG = {
    "mean": lambda g: g.p.mean(),
    "logit mean": lambda g: 1 / (1 + np.exp(-_logit(g.p.to_numpy()).mean())),
    "median": lambda g: g.p.median(),
    "max": lambda g: g.p.max(),
    "top-2 mean": lambda g: g.p.nlargest(2).mean(),
    "power-2": lambda g: (g.p.pow(2).mean()) ** 0.5,
    "power-4": lambda g: (g.p.pow(4).mean()) ** 0.25,
    "plane mean": lambda g: g.groupby("plane").p.mean().mean(),
    "plane max": lambda g: g.groupby("plane").p.mean().max(),
    "noisy-OR": lambda g: 1 - np.prod(1 - g.p.clip(0, 1 - EPS)),
}


def series_preds(ref, finding):
    run, _, planes = ref.partition("@")
    v = pd.read_csv(R + run + "/val_series_preds.csv")
    sub = v.cls.str.contains("|", regex=False)
    v = v[(v.cls == finding) | (sub & (v.cls.str.split("|").str[1].fillna("").map(finding_of) == finding))]
    return v[v.plane.isin(planes.split(","))] if planes else v


def study(v, agg):
    """Per (study, cls) aggregation over series, then max over the per-ROI sub-targets (one cls otherwise)."""
    s = v.groupby(["StudyInstanceUID", "cls"])[["p", "plane"]].apply(agg).rename("p").reset_index()
    return s.groupby("StudyInstanceUID").p.max()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../output/locations/agg_eval.csv")
    a = ap.parse_args()
    st, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(st, holdout_frac=0.1, seed=0)
    S = st.set_index("StudyInstanceUID")
    rows = []
    for c, refs in RECIPE.items():
        G = S.loc[gt_ids, f"gt_{c}"].dropna().round()
        H = (S.loc[hold_ids, f"target_{c}"].dropna() >= 0.5).astype(int)
        V = [series_preds(r, c) for r in refs]
        for name, agg in AGG.items():
            p = sum(study(v, agg).rank(pct=True) for v in V)
            rows.append(dict(finding=c, agg=name, gt=roc_auc_score(G, p.reindex(G.index)),
                             hold=roc_auc_score(H, p.reindex(H.index))))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    for k in ("gt", "hold"):
        tab = df.pivot(index="agg", columns="finding", values=k)[list(RECIPE)]
        tab["mean (12)"] = tab.mean(axis=1)
        if k == "hold":
            tab["mean (11, no syn.)"] = tab.drop(columns=["Synovitis", "mean (12)"]).mean(axis=1)
        print(f"\n== {'58 radiologist studies' if k == 'gt' else 'holdout'}")
        print(tab.loc[list(AGG)].round(3).to_string())


if __name__ == "__main__":
    main()
