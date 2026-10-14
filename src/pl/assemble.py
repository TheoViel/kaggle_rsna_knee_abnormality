"""Out-of-fold pseudo-labels of a weight package's recipe from the k-fold runs (pl.kfold_jobs), run from src/.

    python -m pl.assemble --weights ../output/dataset_8 [--out ../output/pl_dataset_8]

Finds the k-fold runs (logs/crops_run/*/*/args.json with n_folds > 0) of each distinct setup of the package
(pl.kfold_jobs.setups: the seeds of a setup collapse), and for every finding of the recipe:
  - per setup: the out-of-fold study probability (each study scored by the run of its fold; the seeds of a setup
    averaged); a recipe entry "name@Plane1,Plane2" averages the per-series predictions of those planes only;
  - the recipe pseudo-label: the mean probability over the finding's setups (soft labels, calibrated like the
    models), and the rank average over them (the package's own combination, ranks over every training study).
Writes <out>/oof_models.csv (StudyInstanceUID x "<finding>|<setup>"), <out>/pl_mean.csv and <out>/pl_rank.csv
(StudyInstanceUID x 12 findings), <out>/coverage.csv (runs used per setup and fold) and prints the AUC of each
pseudo-label on the 58 radiologist studies (out-of-fold for them too) and on the other studies against the Claude
labels (>= 0.5).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables
from pl.kfold_jobs import SKIP, normalised, setups

CLASSES = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]


def key(a):
    return json.dumps({k: v for k, v in normalised(a).items() if k not in SKIP | {"batch_size"}}, sort_keys=True)


def kfold_runs():
    """{setup key: [(run dir, fold, seed, n_folds)]} of every finished k-fold run."""
    out = {}
    for f in glob.glob("../logs/crops_run/*/*/args.json"):
        a = json.load(open(f))
        d = os.path.dirname(f)
        if not a.get("n_folds") or not os.path.exists(os.path.join(d, "ckpt_final.pt")):
            continue
        out.setdefault(key(a), []).append((d, a["fold"], a["seed"], a["n_folds"]))
    return out


def oof(runs, finding, planes=None):
    """Out-of-fold study probabilities of one setup for one finding (seeds averaged)."""
    per_seed = {}
    for d, fold, seed, _ in runs:
        if planes:
            v = pd.read_csv(os.path.join(d, "val_series_preds.csv"))
            v = v[(v.cls == finding) & v.plane.isin(planes)]
            p = v.groupby("StudyInstanceUID").p.mean()
        else:
            t = pd.read_csv(os.path.join(d, "val_study_preds.csv"))
            t = t.set_index(t.columns[0])
            if finding not in t:
                return None
            p = t[finding]
        per_seed.setdefault(seed, []).append(p)
    # a seed counts only once all its folds are in (otherwise its studies would be partly missing)
    n = max(len(v) for v in per_seed.values())
    full = [pd.concat(v) for v in per_seed.values() if len(v) == n]
    return pd.concat(full, axis=1).mean(axis=1) if full else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="../output/dataset_8")
    ap.add_argument("--out", default="../output/pl_dataset_8")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recipe = json.load(open(os.path.join(a.weights, "recipe.json")))
    S = setups(a.weights)
    name_of = {}                                        # package model -> its setup's first model name
    for m in os.listdir(os.path.join(a.weights, "crops")):
        k = key(json.load(open(os.path.join(a.weights, "crops", m, "args.json"))))
        name_of[m] = next(n for n in S if key(S[n]) == k)
    runs = kfold_runs()
    cov = []
    for n in S:
        r = runs.get(key(S[n]), [])
        cov.append(dict(setup=n, runs=len(r), folds=sorted({f for _, f, _, _ in r}), seeds=sorted({s for _, _, s, _ in r}),
                        n_folds=sorted({k for _, _, _, k in r})))
    cov = pd.DataFrame(cov)
    cov.to_csv(os.path.join(a.out, "coverage.csv"), index=False)
    print(cov.to_string(index=False))
    st, _ = load_tables()
    st = st.set_index("StudyInstanceUID")
    models, mean, rank = {}, {}, {}
    for c in CLASSES:
        entries = recipe.get(c, [])
        cols = []
        for e in dict.fromkeys(name_of[x.split("@")[0]] + ("@" + x.split("@")[1] if "@" in x else "") for x in entries):
            n, _, planes = e.partition("@")
            r = [x for x in runs.get(key(S[n]), []) if x[3] == 4] or runs.get(key(S[n]), [])
            folds = {f for _, f, _, _ in r}
            if not r or len(folds) < max(x[3] for x in r):
                print(f"  {c}: {e} incomplete ({sorted(folds)}), left out")
                continue
            p = oof(r, c, planes.split(",") if planes else None)
            if p is not None:
                models[f"{c}|{e}"] = p
                cols.append(f"{c}|{e}")
        if cols:
            P = pd.concat([models[k] for k in cols], axis=1)
            mean[c] = P.mean(axis=1)
            rank[c] = P.rank(pct=True).mean(axis=1)
    pd.DataFrame(models).rename_axis("StudyInstanceUID").to_csv(os.path.join(a.out, "oof_models.csv"))
    for n_, d in (("pl_mean", mean), ("pl_rank", rank)):
        pd.DataFrame(d).reindex(columns=[c for c in CLASSES if c in d]).rename_axis("StudyInstanceUID").to_csv(os.path.join(a.out, f"{n_}.csv"))
    rows = []
    for c in mean:
        g = st[f"gt_{c}"].dropna().round()
        rest = st.index[st[f"gt_{c}"].isna()] if f"gt_{c}" in st else st.index
        y = (st.loc[rest, f"target_{c}"].dropna() >= 0.5).astype(int)
        rows.append(dict(finding=c, studies=int(mean[c].notna().sum()),
                         gt58=roc_auc_score(g, mean[c].reindex(g.index)) if mean[c].reindex(g.index).notna().all() else np.nan,
                         claude_rest=roc_auc_score(y, mean[c].reindex(y.index)) if mean[c].reindex(y.index).notna().all() else np.nan,
                         models=", ".join(k.split("|")[1] for k in models if k.startswith(c + "|"))))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
