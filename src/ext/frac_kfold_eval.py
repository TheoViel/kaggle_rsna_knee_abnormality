"""Fracture crop sets compared out-of-fold over every training study (k-fold runs of the scheduler); run from src/.

    python -m ext.frac_kfold_eval --a pl_fb0 --b kf16

--a / --b: scheduler job-name prefixes of two k-fold setups (jobs "<prefix>_s<seed>_f<fold>", logs/sched). For each,
the out-of-fold study score is the max over its crops of the series-mean crop probability, seeds 0 and 1
averaged (the same seeds for both setups). Reported:
  - AUC against the Claude fracture labels (>= 0.5) on the studies other than the 58 (report convention);
  - AUC of the definite acute / unknown-acuity report fractures against clear negatives (Claude < 0.1);
  - per report location (output/locations/fracture_locations.json): the share of clear negatives ranked below;
  - AUC on the 58 radiologist studies (out-of-fold too: each fold scores its own 58).
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables


def oof(prefix, keep=("0", "1")):
    log = open("../logs/sched/sched.log").read()
    runs = re.findall(rf"end   {prefix}_s(\d+)_f(\d) rc=0 \d+min crops_run/(\S+)", log)
    seeds = {}
    for s, f, r in runs:
        if s not in keep:
            continue
        v = pd.read_csv(f"../logs/crops_run/{r}/val_series_preds.csv")
        roi = v.groupby(["StudyInstanceUID", "cls"]).p.mean().unstack()
        seeds.setdefault(s, []).append(roi.max(axis=1))
    full = [pd.concat(p) for p in seeds.values() if len(p) == 4]
    print(f"{prefix}: {len(runs)} runs, {len(full)} complete seeds")
    return pd.concat(full, axis=1).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="pl_fb0")
    ap.add_argument("--b", default="kf16")
    a = ap.parse_args()
    st, _ = load_tables()
    S = st.set_index("StudyInstanceUID")
    P = {k: oof(k) for k in (a.a, a.b)}
    idx = P[a.a].index.intersection(P[a.b].index)
    rest = idx[~S.loc[idx, "is_gt"].values]
    gt = idx[S.loc[idx, "is_gt"].values]
    y = (S.loc[rest, "target_Fracture"] >= 0.5).astype(int)
    G = S.loc[gt, "gt_Fracture"].dropna().round()
    F = {s["StudyInstanceUID"]: s for s in json.load(open("../output/locations/fracture_locations.json"))["studies"]}
    pos = [u for u in rest if u in F and F[u]["fracture"] == "yes" and F[u]["acuity"] in ("acute", "unk")]
    neg = rest[(S.loc[rest, "target_Fracture"] < 0.1).values]
    print(f"\nstudies {len(idx)} (58: {len(gt)}); Claude positives {int(y.sum())}; definite report fractures {len(pos)}; clear negatives {len(neg)}")
    print(f"{'':28s} {a.a:>10s} {a.b:>10s}")
    for name, fn in (("AUC vs Claude (non-58)", lambda p: roc_auc_score(y, p.reindex(y.index))),
                     ("definite vs clear negatives", lambda p: roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                                                                             np.r_[p.reindex(pos).values, p.reindex(neg).values])),
                     ("AUC on the 58 (OOF)", lambda p: roc_auc_score(G, p.reindex(G.index)))):
        print(f"{name:28s} {fn(P[a.a]):10.3f} {fn(P[a.b]):10.3f}")
    rows = []
    for u in pos:
        for l in (F[u]["locations"] if isinstance(F[u]["locations"], list) else []):
            rows.append(dict(uid=u, structure=l["structure"], **{k: (P[k][u] > P[k][neg]).mean() for k in (a.a, a.b)}))
    d = pd.DataFrame(rows).drop_duplicates(["uid", "structure"])
    g = d.groupby("structure").agg(n=("uid", "nunique"), **{k: (k, "mean") for k in (a.a, a.b)})
    g["diff"] = g[a.b] - g[a.a]
    print("\nper report location: share of clear negatives ranked below the fracture study")
    print(g.sort_values("n", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
