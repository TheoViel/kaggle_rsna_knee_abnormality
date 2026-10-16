"""Decorrelated OA pseudo-labels from out-of-fold teachers without cross-talk; run from src/.

    PYTHONPATH=. python ../src_bis/make_pl_dec.py --t1 RUN,RUN,.. --t2 ... --t3 ... [--name pl_oadec]

Teachers (5 folds each, src_bis/train_oof.py: folds over the training studies only, the 58 + holdout never trained):
  T1 lateral-only on oa_cmp30_mid, Claude labels;  T2 PF-only on the mk90 crops, Claude labels;
  T3 the mk90 setup (3 findings, pl_round2), for PF and for the medial OA score of the correction.
Per training study, the out-of-fold prediction (the fold that validated it). Fixed in advance (R2's recipe and
pl_round2's mixing weight):
  lateral = rank(T1) - 0.3 x rank(T3 medial)          PF = mean rank(T2, T3) - 0.3 x rank(T3 medial)
re-ranked over the training studies, then mapped to probabilities by quantile: the corrected rank takes the teachers'
own probability at that quantile (T1; mean of T2 and T3), so the ordering is the decorrelated one and the scale is
the teachers'. Target = 0.5 x that + 0.5 x the Claude label. Written as pl_round2 with only the Lateral OA and PF OA
columns of the training studies replaced: output/crops/<name>.csv (main_crops --labels pl:<name>).
"""

import argparse
import os

import numpy as np
import pandas as pd

from knee_cls.data import load_tables, make_splits
from params import OUTPUT_FOLDER

BETA, MIX = 0.3, 0.5


def oof(runs, c, train):
    """Out-of-fold predictions of finding c over the training studies (each run validated one fold)."""
    out = []
    for r in runs:
        t = pd.read_csv(os.path.join("..", "logs", "crops_run", r, "val_study_preds.csv"))
        t = t.set_index(t.columns[0])[c]
        out.append(t[t.index.isin(train)])
    s = pd.concat(out)
    assert s.index.is_unique and len(s) == len(train), (len(s), len(train))
    return s


def quantile_map(score, prob):
    """Values of `prob` re-assigned in the order of `score` (same distribution, decorrelated order)."""
    order = score.rank(method="first").astype(int).to_numpy() - 1
    return pd.Series(np.sort(prob.to_numpy())[order], index=score.index)


def main():
    p = argparse.ArgumentParser()
    for k in ("t1", "t2", "t3"):
        p.add_argument(f"--{k}", required=True, help="comma-separated runs, one per fold")
    p.add_argument("--name", default="pl_oadec")
    a = p.parse_args()
    studies, _ = load_tables()
    train, gt, hold = make_splits(studies, holdout_frac=0.1, seed=0)
    train = set(train)
    t1, t2, t3 = (a.t1.split(","), a.t2.split(","), a.t3.split(","))
    med = oof(t3, "Medial OA", train).rank(pct=True)
    lat_p = oof(t1, "Lateral OA", train)
    pf2, pf3 = oof(t2, "PF OA", train), oof(t3, "PF OA", train)
    lat = (lat_p.rank(pct=True) - BETA * med.reindex(lat_p.index)).rank(pct=True)
    pf = ((pf2.rank(pct=True) + pf3.rank(pct=True).reindex(pf2.index)) / 2 - BETA * med.reindex(pf2.index)).rank(pct=True)
    teacher = {"Lateral OA": quantile_map(lat, lat_p), "PF OA": quantile_map(pf, (pf2 + pf3.reindex(pf2.index)) / 2)}

    claude = studies.set_index("StudyInstanceUID")
    base = pd.read_csv(os.path.join(OUTPUT_FOLDER, "crops", "pl_round2.csv")).set_index("StudyInstanceUID")
    out = base.copy()
    for c, t in teacher.items():
        lab = claude.loc[t.index, f"target_{c}"]
        new = MIX * t + (1 - MIX) * lab.where(lab.notna(), t)       # no Claude label: the teacher alone
        out.loc[new.index, c] = new
    path = os.path.join(OUTPUT_FOLDER, "crops", f"{a.name}.csv")
    out.reset_index().to_csv(path, index=False)

    # Label correlations over the training studies (radiologists, on the 58: 0.32 medial-lateral, 0.37 PF-medial).
    tr = sorted(train)
    srcs = {"Claude": claude.loc[tr, [f"target_{c}" for c in ("Medial OA", "Lateral OA", "PF OA")]].set_axis(
        ["Medial OA", "Lateral OA", "PF OA"], axis=1), "pl_round2": base.loc[tr], a.name: out.loc[tr]}
    print(f"-> {path}\nSpearman over {len(tr)} training studies:")
    for k, v in srcs.items():
        cm = v[["Medial OA", "Lateral OA", "PF OA"]].corr("spearman")
        print(f"  {k:10s} medial-lateral {cm.loc['Medial OA', 'Lateral OA']:.2f}  medial-PF {cm.loc['Medial OA', 'PF OA']:.2f}"
              f"  lateral-PF {cm.loc['Lateral OA', 'PF OA']:.2f}")


if __name__ == "__main__":
    main()
