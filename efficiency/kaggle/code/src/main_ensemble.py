"""Score the average of several runs' validation predictions, run from src/.

    python main_ensemble.py ../logs/cls_run/2026-10-08/2 ../logs/cls_run/2026-10-08/3

Runs must share the split (same `seed` and `holdout_frac`): predictions are
matched by StudyInstanceUID and only studies every run predicted are scored.
Averaging is on the rank-preserving logit scale.
"""

import os
import sys

import numpy as np
import pandas as pd

from knee_cls.train import score
from params import CLASSES


def main(run_dirs):
    dfs = [pd.read_csv(os.path.join(d, "val_preds.csv")).set_index("StudyInstanceUID")
           for d in run_dirs]
    common = sorted(set.intersection(*[set(d.index) for d in dfs]))
    pred_cols = [f"pred_{c}" for c in CLASSES]
    for d, df in zip(run_dirs, dfs):
        m = score(df.loc[common].reset_index())
        print(f"{d:45s} gt_auc {m['gt_auc']:.4f}  hold_auc {m.get('hold_auc', float('nan')):.4f}")
    logit = np.mean([np.log(df.loc[common, pred_cols].clip(1e-6, 1 - 1e-6) /
                            (1 - df.loc[common, pred_cols].clip(1e-6, 1 - 1e-6)))
                     for df in dfs], 0)
    ens = dfs[0].loc[common].copy()
    ens[pred_cols] = 1 / (1 + np.exp(-logit))
    m = score(ens.reset_index())
    print(f"{'ENSEMBLE':45s} gt_auc {m['gt_auc']:.4f}  hold_auc {m.get('hold_auc', float('nan')):.4f}")
    print("  gt  " + "  ".join(f"{c[:8]} {v:.3f}" for c, v in m["gt_per_class"].items()))


if __name__ == "__main__":
    main(sys.argv[1:])
