"""src/main_crops.py with options for the effusion / contusion work; run from src/ (src/ unchanged).

    torchrun ... ../src_bis/train_ec.py [--fs_only] <main_crops.py arguments>

  - group eff_cont = ["Effusion", "Contusion"]: both image-labelled findings of fastMRI+ in one model;
  - --fs_only: train and validate on fluid-sensitive series only (Fluid_Sensitive == 1, every crop set incl. the
    extra one). Bone-marrow edema and joint fluid are bright only on fluid-sensitive (fat-suppressed PD / T2 / STIR)
    series; the main agent's contusion models score 0.914 / 0.928 from the other series, and averaging them over the
    fluid-sensitive series only already lifts contusion 0.950 / 0.956 -> 0.953 / 0.964. Studies without such a series
    are not trained on; at inference they fall back to all series (ec_eval: "@FS" aggregation).
"""

import os
import runpy
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import crops.train as ct  # noqa: E402

ct.GROUPS.update({"eff_cont": ["Effusion", "Contusion"]})
if "--fs_only" in sys.argv:
    sys.argv.remove("--fs_only")
    _load = ct.load_crops

    def load_crops(names):
        arrays, info = _load(names)
        keep = info.Fluid_Sensitive == 1
        print(f"- fs_only {names}: {int(keep.sum())} / {len(info)} series", flush=True)
        return arrays, info[keep].reset_index(drop=True)

    ct.load_crops = load_crops
sys.argv[0] = os.path.join(SRC, "main_crops.py")
runpy.run_path(sys.argv[0], run_name="__main__")
