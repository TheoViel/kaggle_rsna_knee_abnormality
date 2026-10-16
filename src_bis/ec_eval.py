"""Effusion / contusion candidates against b1_c1 and the main agent's models; run from src/.

    PYTHONPATH=. python ../src_bis/ec_eval.py Effusion REF[+REF..] [REF..]      (a ref: run, or seed runs joined by +)

Per candidate (a run, or a seed group = mean rank):
  - AUC on the 58 (radiologists) and on the holdout (report labels, Claude);
  - "+b1": 50/50 rank blend with b1_c1's sealed predictions on the 58 (the notebook's CLASSES_TO_BLEND blend),
    with a paired bootstrap against b1_c1 alone (mean delta, P(delta > 0));
  - Spearman with b1_c1 on the 58 (diversity).
References: b1_c1 alone, the main agent's ef0+ef1 (2026-10-10/27 + 2026-10-13/87) / cc0-2 (2026-10-12/58, 2026-10-13/1, /9).
The LB rejected the main agent's models in that blend, so the bar is: the blend with b1_c1 above b1_c1 alone, with
no loss on the holdout.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oa_eval as E  # noqa: E402

MAIN = {"Effusion": "2026-10-10/27+2026-10-13/87", "Contusion": "2026-10-12/58+2026-10-13/1+2026-10-13/9"}
B1 = pd.read_csv(os.path.join("..", "input", "b1_c1_sealed_58_predictions.csv")).set_index("StudyInstanceUID")


TS = pd.read_csv(os.path.join("..", "input", "train_series.csv")).set_index("SeriesInstanceUID").Fluid_Sensitive


def study_fs(run, c):
    """Study probability from the fluid-sensitive series only (all series when a study has none)."""
    v = pd.read_csv(os.path.join(E.run_dir(run), "val_series_preds.csv"))
    v = v[v.cls == c]
    allp = v.groupby("StudyInstanceUID").p.mean()
    fs = v[v.SeriesInstanceUID.map(TS).eq(1)].groupby("StudyInstanceUID").p.mean()
    return fs.reindex(allp.index).fillna(allp)


def group(ref, c):
    """'run+run+..' (seed group, mean rank); suffix '@FS': study scores from fluid-sensitive series only."""
    ref, _, agg = ref.partition("@")
    return E.rank_mean([study_fs(r, c) if agg == "FS" else E.study(r, c) for r in ref.split("+")])


def boot58(p_new, p_old, c, n=2000, seed=0):
    st, gt, _ = E.tables()
    y = (st.loc[gt, f"gt_{c}"].to_numpy() >= 0.5).astype(int)
    a, b = p_new.reindex(gt).to_numpy(), p_old.reindex(gt).to_numpy()
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(gt), len(gt))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    return d.mean(), (d > 0).mean()


def report(c, refs):
    st, gt, _ = E.tables()
    b1 = B1.loc[gt, c].rank(pct=True)
    y = (st.loc[gt, f"gt_{c}"] >= 0.5).astype(int)
    print(f"{c}: {y.sum()} positives / {len(y)} on the 58;  b1_c1 alone {roc_auc_score(y, b1):.4f}")
    for ref in [MAIN[c], MAIN[c] + "@FS"] + refs:
        p = group(ref, c)
        g = p.reindex(gt).rank(pct=True)
        blend = (g + b1) / 2
        m, q = boot58(blend, b1, c)
        tag = "MAIN " if ref.split("@")[0] == MAIN[c] else ""
        print(f"  {tag}{ref:40s} {E.fmt(E.auc(p, c))}   +b1 {roc_auc_score(y, blend):.4f} ({m:+.4f}, P {q:.2f})"
              f"   corr b1 {spearmanr(g, b1)[0]:.2f}")


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2:])
