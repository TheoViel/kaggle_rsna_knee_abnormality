"""Head-to-head on OAI effusion (MOAKS, image-graded): b1_c1 (team runtime on the OAI root) vs our models vs blends.

    PYTHONPATH=. python ../src_bis/oai_h2h.py TEAM_WORK_DIR REF [REF ...]      (run from src/; REF: run, or a+b seeds)

TEAM_WORK_DIR: team_infer.py --work on the root of src_bis/make_oai_root.py (300 knees: 100 each of MOAKS effusion
grade >= 2, mild, 0). b1_c1's effusion = its baseline_submission.csv column (B1 0.9 + C1 0.1 after member ranks).
Our models: their predictions on the OAI capsule crops (oai_cap_e15_d24_study_preds.csv, ext_bench.py), the same
knees. Blend = the notebook's: 50/50 rank average with b1_c1. Columns: >=2 vs 0, >=2 vs mild, any (mild+) vs 0.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_bench as X  # noqa: E402

work, refs = sys.argv[1], sys.argv[2:]
b1 = pd.read_csv(os.path.join(work, "baseline_submission.csv") if os.path.isdir(work) else work).set_index("StudyInstanceUID").Effusion
lab = X.OAI.Effusion.reindex(b1.index)
cuts = {">=2 vs 0": lab.isin([0, 1]), ">=2 vs mild": lab.isin([0.25, 1]), "any vs 0": lab.isin([0, 0.25, 1])}
pos = {">=2 vs 0": lab == 1, ">=2 vs mild": lab == 1, "any vs 0": lab > 0}


def row(name, p):
    p = p.reindex(b1.index)
    return f"{name:44s} " + "  ".join(f"{k} {roc_auc_score(pos[k][m].astype(int), p[m]):.3f}"
                                      for k, m in cuts.items() if p[m].notna().all())


print(row("b1_c1", b1))
for ref in refs:
    p = X.group(ref, X.SETS["oai"], "Effusion", "2")
    print(row(ref, p))
    print(row("   + b1_c1 (50/50 rank)", (p.reindex(b1.index).rank(pct=True) + b1.rank(pct=True)) / 2))
