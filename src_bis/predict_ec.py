"""crops.predict with the extra groups of src_bis (train_ec.py, train_group.py); run from src/ (src/ unchanged)."""
import os
import runpy
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
import crops.train as ct  # noqa: E402

ct.GROUPS.update({"eff_cont": ["Effusion", "Contusion"], "pf_oa": ["PF OA"], "lateral_oa": ["Lateral OA"],
                  "medial_oa": ["Medial OA"]})
sys.argv[0] = "crops.predict"
runpy.run_module("crops.predict", run_name="__main__")
