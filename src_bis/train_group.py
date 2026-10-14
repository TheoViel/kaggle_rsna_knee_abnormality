"""src/main_crops.py with extra single-finding OA groups, run from src/ (src/ itself unchanged).

    cd src && torchrun --nproc_per_node=2 ../src_bis/train_group.py --group pf_oa ... (main_crops.py arguments)

Groups added: pf_oa = ["PF OA"], lateral_oa = ["Lateral OA"], medial_oa = ["Medial OA"]: one finding per model, so
the encoder cannot share "arthritic knee" features learned from the other compartments' targets.
"""

import os
import runpy
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import crops.train as ct  # noqa: E402

ct.GROUPS.update({"pf_oa": ["PF OA"], "lateral_oa": ["Lateral OA"], "medial_oa": ["Medial OA"]})
sys.argv[0] = os.path.join(SRC, "main_crops.py")
runpy.run_path(sys.argv[0], run_name="__main__")
