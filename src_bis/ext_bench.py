"""External radiologist-labelled benchmark for effusion / contusion models; run from src/.

    PYTHONPATH=. python ../src_bis/ext_bench.py [--gpu 2] REF [REF ...]     (a ref: run, or seed runs joined by +)

The 58 cannot separate effusion models (b1_c1 0.989 = ~9 misordered pairs of 805) and has 19 contusion positives.
These external sets are labelled from the images, like the 58 and the LB, and models trained on our data never saw
them:
  fastMRI+ (1,172 knees, one coronal PD series each; output/fastmri/labels_graded.csv):
      effusion 142 / 1,030; contusion (fracture / contusion / dislocation boxes) 119 / 1,053;
  OAI baseline (2,400 knees, sagittal + coronal IW-TSE / DESS / MPR; output/oai/labels_moaks00_v4.csv):
      effusion MOAKS grade >= 2 (326) vs 0 (713); "mild" also scores grade >= 2 vs mild (617).
fastMRI is split into two stratified halves (output/fastmri/halves.csv): candidates trained with fastMRI use half A
only and are compared on half B ("fmB": effusion 72 / 515, contusion 60 / 527).
Crops: fm_cap_e15_d24 / oai_cap_e15_d24 (capsule +15%, 24 frames, 160 px). Predictions: crops.predict --crops, cached
as <run dir>/<crop set>_study_preds.csv. Models trained on OAI (OAI-only, joint) are leaky on the OAI columns.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNS = os.path.join("..", "logs", "crops_run")
SETS = {"fm": "fm_cap_e15_d24", "oai": "oai_cap_e15_d24"}
FM = pd.read_csv(os.path.join("..", "output", "fastmri", "labels_graded.csv")).set_index("StudyInstanceUID")
HALF = pd.read_csv(os.path.join("..", "output", "fastmri", "halves.csv")).set_index("StudyInstanceUID").half
OAI = pd.read_csv(os.path.join("..", "output", "oai", "labels_moaks00_v4.csv")).set_index("StudyInstanceUID")


def preds(run, crop_set, gpu):
    path = os.path.join(RUNS, run, f"{crop_set}_study_preds.csv")
    if not os.path.exists(path):
        subprocess.run([sys.executable, "../src_bis/predict_ec.py", "--runs", f"../{run}", "--crops", crop_set],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(path).set_index("StudyInstanceUID")


def group(ref, crop_set, c, gpu):
    cols = [preds(r, crop_set, gpu) for r in ref.split("+")]
    cols = [t[c] for t in cols if c in t]
    return sum(s.rank(pct=True) for s in cols) / len(cols) if cols else None


def auc(y, p):
    i = y.index.intersection(p.index)
    return roc_auc_score(y[i], p[i]) if len(i) and y[i].nunique() == 2 else np.nan


def bench(ref, gpu):
    out = {}
    for c in ("Effusion", "Contusion"):
        p = group(ref, SETS["fm"], c, gpu)
        if p is not None:
            y = (FM[c] >= 0.5).astype(int)
            out[f"fm {c[:3]}"] = auc(y, p)
            out[f"fmB {c[:3]}"] = auc(y[HALF.reindex(y.index) == "B"], p)     # half B: never trained on
    p = group(ref, SETS["oai"], "Effusion", gpu)
    if p is not None:
        e = OAI.Effusion
        out["oai Eff"] = auc((e[e.isin([0, 1])] == 1).astype(int), p)
        out["oai Eff vs mild"] = auc((e[e.isin([0.25, 1])] == 1).astype(int), p)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--gpu", default="2")
    a.add_argument("refs", nargs="+")
    a = a.parse_args()
    for ref in a.refs:
        r = bench(ref, a.gpu)
        print(f"{ref:44s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)


if __name__ == "__main__":
    main()
