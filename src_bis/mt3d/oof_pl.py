"""Pseudo-labels for the 4 findings without an ROI from out-of-fold teachers, run from src_bis/.

    python -m mt3d.oof_pl --runs 2026-10-12/35,... --out ../output/bis/pl_extra_codex_oof.csv [--w 0.5]

Each run is a `main_mt3d.py --oof_fold k` teacher: trained without fold k of the training studies, it
predicted that fold (with the validation studies) in val_study_preds.csv. A training study's teacher
prediction comes from the run that held out its fold; validation studies get the mean over the runs (never
used for training). Target = (1 - w) x Codex label + w x teacher prediction (Codex alone where the teacher has
no prediction, e.g. a study without sagittal / coronal series).
"""

import argparse
import json

import numpy as np
import pandas as pd

from params import INPUT_FOLDER

EXTRA = ["Effusion", "Synovitis", "Contusion", "Fracture"]
MT = "../logs/bis/mt3d/"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--w", type=float, default=0.5)
    a = p.parse_args()
    from knee_cls.data import load_tables, make_splits
    studies, _ = load_tables()
    _, gt, hold = make_splits(studies, 0.1, 0)
    val = set(gt) | set(hold)
    oof, val_preds = [], []
    for r in a.runs.split(","):
        args = json.load(open(f"{MT}{r}/args.json"))
        t = pd.read_csv(f"{MT}{r}/val_study_preds.csv", index_col=0)[EXTRA]
        oof.append(t[~t.index.isin(val)])                  # this run's held-out fold
        val_preds.append(t[t.index.isin(val)])
        print(f"{r}: fold {args['oof_fold']}, {(~t.index.isin(val)).sum()} held-out studies")
    teacher = pd.concat(oof + [pd.concat(val_preds).groupby(level=0).mean()])
    assert not teacher.index.duplicated().any(), "a study in two folds"
    codex = pd.concat([pd.read_csv(f"{INPUT_FOLDER}codex-labels_final_{n}.csv") for n in (52, 4349)]).set_index("StudyInstanceUID")[EXTRA]
    teacher = teacher.reindex(codex.index)
    pl = ((1 - a.w) * codex + a.w * teacher).fillna(codex)
    pl.reset_index().to_csv(a.out, index=False)
    print(f"-> {a.out}: {len(pl)} studies, teacher on {teacher.notna().all(1).sum()}; mean targets {pl.mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
