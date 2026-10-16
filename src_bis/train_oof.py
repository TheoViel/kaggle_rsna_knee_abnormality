"""src/main_crops.py k-fold teachers that never see the evaluation studies; run from src/ (src/ unchanged).

    cd src && torchrun --nproc_per_node=2 ../src_bis/train_oof.py --n_folds 5 --fold k ... (main_crops.py arguments)

main_crops' k-fold mode (out-of-fold pseudo-labels) spreads every study over the folds, the 58 radiologist studies
and the holdout included: the teachers of the other folds train on their labels, and pseudo-labels made from them
leak the evaluation labels into the students. Here the folds cover the training studies only (make_splits' train
ids, same seeded shuffle as make_folds); the 58 and the holdout are put in every fold's validation set, so they are
predicted by each fold model and trained on by none. Each run's val_study_preds.csv = the out-of-fold predictions of
its fold + the 58 + the holdout. Also adds the single-finding groups (as train_group.py).
"""

import os
import runpy
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import crops.train as ct  # noqa: E402
import knee_cls.data as kd  # noqa: E402

ct.GROUPS.update({"pf_oa": ["PF OA"], "lateral_oa": ["Lateral OA"], "medial_oa": ["Medial OA"]})
FOLD = int(sys.argv[sys.argv.index("--fold") + 1])


def make_folds(studies, n_folds, seed=42):
    train, gt, hold = kd.make_splits(studies, holdout_frac=0.1, seed=0)
    rng = np.random.default_rng(seed)
    ids = np.sort(np.asarray(train))
    ids = ids[rng.permutation(len(ids))]
    out = {s: i % n_folds for i, s in enumerate(ids)}
    out.update({s: FOLD for s in list(gt) + list(hold)})       # always validation, never training
    return out


kd.make_folds = make_folds
sys.argv[0] = os.path.join(SRC, "main_crops.py")
runpy.run_path(sys.argv[0], run_name="__main__")
