"""Noisy-OR training targets from an OAI-only teacher, run from src/ (the synovitis recipe of dataset_6, generalised).

    python -m oai.teacher_pl --teachers ../2026-10-17/30,../2026-10-17/31 --name oai3_noisyor

The teachers are crop models trained only on OAI MOAKS gradings (main_crops.py --extra_only True), so their predictions
on the competition studies never saw a competition label. For each finding a teacher predicts
(<run>/all_study_preds.csv, crops.predict):

    target = 1 - (1 - base) * (1 - rank_pct(teacher))

base = the label the competition models already train on: the user's round 2 pseudo-labels (pl_round2) for the 8 ROI
findings, the Claude report label for effusion and synovitis (as the adopted synovitis targets). The teacher's prediction is turned into a percentile rank
over all studies first, as for synovitis (output/synovitis_share/README.md). Several teachers predicting a finding are
rank-averaged. Writes output/crops/<name>.csv (StudyInstanceUID x the 8 ROI findings, effusion, synovitis), for
main_crops.py --labels pl:<name>.
"""
import argparse
import os

import pandas as pd

from knee_cls.data import load_tables
from main_crops import training_targets
from params import OUTPUT_FOLDER
from roi.extract import ROI_CLASSES

RUNS = os.path.join(OUTPUT_FOLDER, "..", "logs", "crops_run", "2026-10-09")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teachers", required=True, help="teacher runs, relative to logs/crops_run/2026-10-09 as crops.predict")
    p.add_argument("--name", required=True)
    p.add_argument("--findings", default="", help="restrict to these findings (comma-separated); default: all predicted")
    a = p.parse_args()
    st, _ = load_tables()
    ids = st.StudyInstanceUID
    cols = list(ROI_CLASSES) + ["Effusion", "Synovitis"]
    pl2 = training_targets("pl:pl_round2", st, cols)
    mean = training_targets("claude", st, cols)
    base = pd.DataFrame({c: (pl2[c].reindex(ids) if len(pl2[c]) else pd.Series(index=ids, dtype=float)).fillna(
        mean[c].reindex(ids)) for c in cols}, index=ids)
    preds = [pd.read_csv(os.path.join(RUNS, t, "all_study_preds.csv")).set_index("StudyInstanceUID").reindex(ids)
             for t in a.teachers.split(",")]
    keep = set(a.findings.split(",")) if a.findings else None
    # Only findings MOAKS grades: a teacher's head for an ungraded finding (MCL) is untrained noise.
    graded = pd.read_csv(os.path.join(OUTPUT_FOLDER, "oai", "labels_moaks_all_v4.csv")).set_index("StudyInstanceUID")
    graded = {c for c in cols if c in graded and graded[c].notna().any()}
    out = base.copy()
    for c in cols:
        ranks = [p_[c].rank(pct=True) for p_ in preds if c in p_ and p_[c].notna().any()]
        if not ranks or (keep and c not in keep) or c not in graded:
            continue
        r = sum(ranks) / len(ranks)
        out[c] = 1 - (1 - base[c]) * (1 - r.fillna(r.median()))
        print(f"{c}: {len(ranks)} teacher(s); mean target {base[c].mean():.3f} -> {out[c].mean():.3f}")
    out.index.name = "StudyInstanceUID"
    path = os.path.join(OUTPUT_FOLDER, "crops", f"{a.name}.csv")
    out.to_csv(path)
    print(f"-> {path}: {len(out)} studies")


if __name__ == "__main__":
    main()
