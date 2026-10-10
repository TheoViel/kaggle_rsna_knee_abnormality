"""Blend ROI crop models with the global classifier, scored on all 12 findings, run from src/.

    python -m crops.blend --runs 2026-10-09/3,2026-10-09/5

For each ROI finding, among the given runs that predict it, the one with the
best holdout (Claude) AUC is used; its study probabilities are rank-averaged
with the global classifier ensemble's (output/global_ens_val_preds.csv) at weight
w (w = 1: ROI model alone). The four findings without an ROI (Effusion,
Synovitis, Contusion, Fracture) keep the global prediction.

w is chosen per finding on the holdout Claude AUC -- not on the 58 radiologist
studies, which are the number reported and too few to tune on.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables
from params import CLASSES

RUNS_DIR = "../logs/crops_run/"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def rank(s):
    return s.rank(pct=True)


def auc(y, p):
    ok = y.notna()
    return roc_auc_score(y[ok], p[ok]) if y[ok].nunique() == 2 else np.nan


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--runs", required=True)
    a.add_argument("--global_preds", default="../output/global_ens_val_preds.csv")
    args = a.parse_args()

    studies, _ = load_tables()
    st = studies.set_index("StudyInstanceUID")
    glob_ = pd.read_csv(args.global_preds).set_index("StudyInstanceUID")
    idx = glob_.index
    gt = st.loc[idx, [f"gt_{c}" for c in CLASSES]]
    is_gt = gt.notna().all(axis=1)
    claude = (st.loc[idx, [f"target_{c}" for c in CLASSES]] >= 0.5).astype(int)

    runs = {r: pd.read_csv(os.path.join(RUNS_DIR, r, "val_study_preds.csv"), index_col=0) for r in args.runs.split(",")}
    rows = []
    for c in CLASSES:
        g = glob_[f"pred_{c}"]
        y_gt, y_hold = gt.loc[is_gt, f"gt_{c}"], claude.loc[~is_gt, f"target_{c}"]
        best = None
        for r, tab in runs.items():
            if c not in tab:
                continue
            roi = tab[c].reindex(idx)
            h = auc(y_hold, roi[~is_gt].fillna(roi.mean()))
            if best is None or h > best[1]:
                best = (r, h, roi.fillna(roi.mean()))
        if best is None:
            rows.append(dict(cls=c, run="global", w=0.0, gt=auc(y_gt, g[is_gt]), hold=auc(y_hold, g[~is_gt]),
                             gt_global=auc(y_gt, g[is_gt]), hold_global=auc(y_hold, g[~is_gt])))
            continue
        r, _, roi = best
        cand = []
        for w in WEIGHTS:
            p = w * rank(roi) + (1 - w) * rank(g)
            cand.append((auc(y_hold, p[~is_gt]), w, auc(y_gt, p[is_gt])))
        h, w, gt_auc = max(cand)
        rows.append(dict(cls=c, run=r, w=w, gt=gt_auc, hold=h, gt_global=auc(y_gt, g[is_gt]),
                         hold_global=auc(y_hold, g[~is_gt]), gt_roi_only=cand[-1][2], hold_roi_only=cand[-1][0]))
    t = pd.DataFrame(rows).set_index("cls")
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print(f"\nmacro over 12: radiologists {t["gt"].mean():.4f} (global alone {t["gt_global"].mean():.4f})   "
          f"holdout Claude {t["hold"].mean():.4f} (global alone {t["hold_global"].mean():.4f})")


if __name__ == "__main__":
    main()
