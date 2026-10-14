"""OA evaluation: runs alone and in the current OA recipe, on the 58 radiologist studies and the holdout; run from src/.

    PYTHONPATH=. python ../src_bis/oa_eval.py --run 2026-10-14/3 [--run bis:2026-10-14/2 ...]

Run refs: "<date>/<n>" (logs/crops_run), "bis:<date>/<n>" (logs/bis/crops_run), optional "@Plane1,Plane2" to average
over those planes' series only. Recipe = the validation counterparts of the submitted OA entries:
  Medial OA: 75 (all), 33 (mk89, OA 224 px MOAKS), 17 (mk90, OA + axial MOAKS); Lateral OA: 33; PF OA: 17.
Per finding (58 / holdout): the run alone, the recipe, the recipe + run (one more vote), and for single-model
findings the recipe with its model replaced. Also the medial-lateral OA prediction correlation (rank, 58 + holdout):
the cross-talk measure (radiologist labels: 0.32; recipe models: ~0.8).
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables, make_splits

OA = ["Medial OA", "Lateral OA", "PF OA"]
RECIPE = {"Medial OA": ["2026-10-09/75", "2026-10-11/33", "2026-10-11/17"], "Lateral OA": ["2026-10-11/33"],
          "PF OA": ["2026-10-11/17"]}
_st = None
_cache = {}


def tables():
    global _st
    if _st is None:
        st, _ = load_tables()
        _, gt, hold = make_splits(st, 0.1, 0)
        _st = (st.set_index("StudyInstanceUID"), list(gt), list(hold))
    return _st


def run_dir(run):
    if run.startswith("bis:"):
        return os.path.join("..", "logs", "bis", "crops_run", run[4:])
    return os.path.join("..", "logs", "crops_run", run)


def study(ref, c):
    """Study probability of finding c from a run ref (series mean; '@planes' restricts the series)."""
    key = (ref, c)
    if key not in _cache:
        run, _, planes = ref.partition("@")
        d = run_dir(run)
        if planes:
            v = pd.read_csv(os.path.join(d, "val_series_preds.csv"))
            v = v[(v.cls == c) & v.plane.isin(planes.split(","))]
            _cache[key] = v.groupby("StudyInstanceUID").p.mean()
        else:
            t = pd.read_csv(os.path.join(d, "val_study_preds.csv"))
            t = t.set_index(t.columns[0])
            _cache[key] = t[c] if c in t else pd.Series(dtype=float)
    return _cache[key]


def rank_mean(series):
    return sum(s.rank(pct=True) for s in series) / len(series)


def combine(refs, c, minus=None):
    """Rank average of the refs for finding c; minus=(refs_other, finding_other, beta) subtracts beta x the other
    finding's combined score (ranks), then re-ranks."""
    p = rank_mean([study(r, c) for r in refs])
    if minus:
        o_refs, o_c, beta = minus
        p = (p - beta * rank_mean([study(r, o_c) for r in o_refs])).rank(pct=True)
    return p


def auc(p, c):
    st, gt, hold = tables()
    g, h = p.reindex(gt), p.reindex(hold)
    a58 = roc_auc_score(st.loc[gt, f"gt_{c}"], g.fillna(g.median()))
    ah = roc_auc_score(st.loc[hold, f"target_{c}"] >= 0.5, h.fillna(h.median()))
    return a58, ah


def crosstalk(p_med, p_lat):
    st, gt, hold = tables()
    idx = gt + hold
    return float(p_med.reindex(idx).rank().corr(p_lat.reindex(idx).rank()))


def fmt(x):
    return f"{x[0]:.3f} / {x[1]:.3f}"


def report(runs):
    rows = []
    rec = {c: auc(combine(RECIPE[c], c), c) for c in OA}
    rows.append(["recipe"] + [fmt(rec[c]) for c in OA] +
                [fmt(np.mean([rec[c] for c in OA], 0))])
    for r in runs:
        alone = {c: auc(study(r, c), c) for c in OA}
        add = {c: auc(combine(RECIPE[c] + [r], c), c) for c in OA}
        rep = {c: auc(study(r, c), c) if len(RECIPE[c]) == 1 else add[c] for c in OA}
        ct = crosstalk(study(r, "Medial OA"), study(r, "Lateral OA"))
        rows.append([f"{r} alone (MOA-LOA corr {ct:.2f})"] + [fmt(alone[c]) for c in OA] +
                    [fmt(np.mean([alone[c] for c in OA], 0))])
        rows.append([f"recipe + {r}"] + [fmt(add[c]) for c in OA] + [fmt(np.mean([add[c] for c in OA], 0))])
        rows.append([f"recipe, single models -> {r}"] + [fmt(rep[c]) for c in OA] +
                    [fmt(np.mean([rep[c] for c in OA], 0))])
    print(pd.DataFrame(rows, columns=["", *OA, "mean"]).set_index("").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    a = ap.parse_args()
    report(a.run)


def bootstrap_delta(p_new, p_old, c, n=2000, seed=0):
    """Paired bootstrap of AUC(p_new) - AUC(p_old) over studies, on the 58 and on the holdout ->
    {(set): (mean delta, 5th percentile, 95th percentile, P(delta > 0))}."""
    st, gt, hold = tables()
    rng = np.random.default_rng(seed)
    out = {}
    for name, ids, col in (("58", gt, f"gt_{c}"), ("holdout", hold, f"target_{c}")):
        y = st.loc[ids, col].to_numpy()
        y = (y >= 0.5).astype(int)
        a = p_new.reindex(ids).fillna(p_new.median()).to_numpy()
        b = p_old.reindex(ids).fillna(p_old.median()).to_numpy()
        d = []
        for _ in range(n):
            i = rng.integers(0, len(ids), len(ids))
            if y[i].min() == y[i].max():
                continue
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
        d = np.array(d)
        out[name] = (d.mean(), np.percentile(d, 5), np.percentile(d, 95), (d > 0).mean())
    return out
