"""Pseudo-labels for the ROI findings from teacher crop models, run from src/.

    python -m crops.pseudo_labels --teachers teachers.json

teachers.json: {finding: [crop run dir, ...]}, the 5 runs with the best holdout score for the finding (holdout runs,
not fullfits). Each run must have predicted every study first (`python -m crops.predict`, all_study_preds.csv; on
the training studies these are in-sample predictions, so the pseudo-labels are partly self-distillation).

Writes output/crops/
  pl_round1.csv  per study and finding, the mean of its teachers' probabilities;
  pl_round2.csv  0.5 x pl_round1 + 0.5 x the mean of the Claude and Codex report labels (weight fixed a priori),
                 what `main_crops.py --labels pl:pl_round2` trains on.
"""

import argparse
import json
import os

import pandas as pd

from crops.train import CROP_DIR
from knee_cls.data import load_tables
from main_crops import training_targets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teachers", required=True)
    a = p.parse_args()
    teachers = json.load(open(a.teachers))
    studies, _ = load_tables()
    r1 = pd.DataFrame({c: pd.concat([pd.read_csv(os.path.join(d, "all_study_preds.csv"))
                                     .set_index("StudyInstanceUID")[c] for d in runs], axis=1).mean(axis=1)
                       for c, runs in teachers.items()}).reindex(studies.StudyInstanceUID)
    mean = training_targets("mean", studies, list(teachers))
    r2 = pd.DataFrame({c: 0.5 * r1[c] + 0.5 * mean[c].reindex(r1.index) for c in teachers})
    for name, df in (("pl_round1", r1), ("pl_round2", r2)):
        df.to_csv(os.path.join(CROP_DIR, f"{name}.csv"))
        print(f"-> {os.path.join(CROP_DIR, name + '.csv')}: {df.shape[0]} studies x {list(df.columns)}")


if __name__ == "__main__":
    main()
