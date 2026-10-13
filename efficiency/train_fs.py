"""src/main_crops.py restricted to fluid-sensitive series (the efficiency pipeline reads only those), run from src/.

    cd src && torchrun --nproc_per_node=2 ../efficiency/train_fs.py --group oa --crops f_s1n8 \
        --planes Sagittal,Coronal --init ../logs/crops_run/2026-10-11/8 --labels pl:pl_round2 ...

Same training as main_crops.py (unchanged, run as __main__); crops.train.load_crops is wrapped so every crop set
keeps only its Fluid_Sensitive == 1 series, for training and validation. Logs go to logs/bis/crops_run/.
"""

import os
import runpy
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import crops.train as ct  # noqa: E402

_load = ct.load_crops


def load_fs(names):
    arrays, info = _load(names)
    if "Fluid_Sensitive" in info:
        info = info[info.Fluid_Sensitive == 1].reset_index(drop=True)
    return arrays, info


ct.load_crops = load_fs
if not any(a.startswith("--log_path") for a in sys.argv):
    sys.argv += ["--log_path", "../logs/bis/crops_run/"]
sys.argv[0] = os.path.join(SRC, "main_crops.py")
runpy.run_path(sys.argv[0], run_name="__main__")
