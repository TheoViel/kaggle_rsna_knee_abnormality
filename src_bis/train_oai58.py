"""src/main_crops.py trained on OAI + the 58 radiologist studies only; run from src/ (src/ itself unchanged).

    cd src && torchrun --nproc_per_node=2 ../src_bis/train_oai58.py --extra_only True --fullfit True \
        --extra_crops oai_s1n8 --extra_labels ../output/oai/labels_moaks00_v4.csv --crops f_s1n8 ... [--gt_repeat k]

The OAI-only models (`--extra_only True`: OAI crops and MOAKS labels, none of our studies) plus our 58
radiologist-labelled studies, with their radiologist labels: `--fullfit True` sets the 58's targets to their gt_
labels, and the extra crop set's table gets the 58's rows of `--crops` appended (main_crops trains on that table
alone with --extra_only). `--gt_repeat k` repeats those rows k times (default 1: each series counts once, ~2.6% of
the items next to 9,651 OAI series). The other training studies are not used; the holdout stays out of training and
is the only clean validation (the 58 are leaky).
"""

import os
import runpy
import sys

import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import crops.train as ct  # noqa: E402
from knee_cls.data import load_tables, make_splits  # noqa: E402


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


repeat = int(arg("--gt_repeat", 1))
if "--gt_repeat" in sys.argv:
    i = sys.argv.index("--gt_repeat")
    del sys.argv[i:i + 2]
assert arg("--extra_only") == "True" and arg("--fullfit") == "True", "needs --extra_only True --fullfit True"
ours, extra_name = arg("--crops"), arg("--extra_crops")
_load = ct.load_crops


def load_crops(names):
    arrays, info = _load(names)
    if names != extra_name:
        return arrays, info
    studies, _ = load_tables()
    _, gt_ids, _ = make_splits(studies, holdout_frac=0.1, seed=0)
    o_arrays, o_info = _load(ours)
    rows = o_info[o_info.StudyInstanceUID.isin(set(gt_ids))]
    print(f"- OAI + 58: {len(info)} {names} series + {len(rows)} series of the {rows.StudyInstanceUID.nunique()} "
          f"radiologist studies ({ours}) x {repeat}", flush=True)
    return {**arrays, **o_arrays}, pd.concat([info] + [rows] * repeat, ignore_index=True)


ct.load_crops = load_crops
sys.argv[0] = os.path.join(SRC, "main_crops.py")
runpy.run_path(sys.argv[0], run_name="__main__")
