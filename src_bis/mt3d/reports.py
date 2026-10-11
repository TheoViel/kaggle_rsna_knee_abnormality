"""Report embeddings as an auxiliary target (`main_mt3d.py --w_rep`).

The radiology report of every study was embedded once with a multilingual sentence encoder
(intfloat/multilingual-e5-base, "passage: " prefix, mean pooling, L2-normalised; reports are in
English, German, Dutch, Spanish, Greek, ...): output/bis/report_emb_e5.npy, one row per study of
input/train.csv (report_emb_e5.csv). A linear probe on these 768-d vectors recovers the Claude labels
on the holdout at AUC 0.69 (MCL) to 0.96 (synovitis): they carry the findings, and the rest of the report.

Targets: the embeddings of the training studies, reduced to `dim` principal components fitted on those
studies only (validation reports are never used), L2-normalised. The model predicts them from its
study token with a cosine loss.
"""

import os

import numpy as np
import pandas as pd

from params import OUTPUT_FOLDER

EMB = os.path.join(OUTPUT_FOLDER, "bis", "report_emb_e5")


def report_targets(train_ids, dim=64):
    e = np.load(EMB + ".npy")
    ids = pd.read_csv(EMB + ".csv").StudyInstanceUID
    tr = ids.isin(set(train_ids)).to_numpy()
    mu = e[tr].mean(0)
    _, _, vt = np.linalg.svd(e[tr] - mu, full_matrices=False)
    z = (e[tr] - mu) @ vt[:dim].T
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return pd.DataFrame(z, index=ids[tr].values)
