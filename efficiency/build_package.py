"""Kaggle datasets for the efficiency notebook, run from efficiency/.

    python build_package.py [--jc1 ../logs/bis/crops_run/2026-10-12/3]

kaggle/code/     src/ (python files), efficiency's pipeline.py / fast_dicom.py / fast_consensus.py, and
                 dynamic_network_architectures (the ROI U-Net's architecture: no pip install in the notebook)
kaggle/weights/  side/, roi/, crops/<model>/ (args.json + ckpt_final.pt in float16) and recipe.json: only the models
                 the recipe uses (fullfit weights of output/dataset_2_moaks_mrnet_men_frac + the joint4 Codex fullfit)
"""

import argparse
import glob
import json
import os
import shutil

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUBMITTED = os.path.join(ROOT, "output", "dataset_2_moaks_mrnet_men_frac")
RECIPE = {"ACL": ["mr85"], "MCL": ["pl86"], "Medial Meniscus": ["cafax87"], "Lateral Meniscus": ["cafax87"],
          "Medial OA": ["mk90"], "Lateral OA": ["mk90"], "PF OA": ["mk90"], "Baker's": ["91"],
          "Effusion": ["jc1"], "Synovitis": ["jc1"], "Contusion": ["jc1"], "Fracture": ["jc1"]}
# Optional (recipe_plus.json): + mk87 on the menisci and the main agent's joint model j28 (fullfit of 2026-10-10/27)
# on the 4 joint findings. Validation: 58 0.9451 / holdout 0.9421 vs 0.9447 / 0.9391, +6.7 H100 ms per study.
PLUS = {c: v + (["mk87"] if "Meniscus" in c else []) + (["j28"] if v == ["jc1"] else []) for c, v in RECIPE.items()}
EXTRA_SRC = {"j28": os.path.join(ROOT, "output", "dataset_3", "crops", "j28")}


def half_copy(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(os.path.join(src_dir, "args.json"), dst_dir)
    sd = torch.load(os.path.join(src_dir, "ckpt_final.pt"), map_location="cpu")
    done = {}                                   # aliased entries (nnU-Net: conv.weight == all_modules.0.weight) stay shared
    for k, v in sd.items():
        if v.is_floating_point():
            key = (v.data_ptr(), v.shape, v.stride())
            sd[k] = done.setdefault(key, v.half())
    torch.save(sd, os.path.join(dst_dir, "ckpt_final.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jc1", default=os.path.join(ROOT, "logs", "bis", "crops_run", "2026-10-12", "3"))
    p.add_argument("--out", default=os.path.join(HERE, "kaggle"))
    a = p.parse_args()
    args = json.load(open(os.path.join(a.jc1, "args.json")))
    assert args["group"] == "joint4" and args["labels"] == "codex", args
    print("jc1:", a.jc1, "fullfit" if args.get("fullfit") else "NOT FULLFIT (validation run)")

    code = os.path.join(a.out, "code")
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(code, "src"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", "*.npy", "*.csv", "*.pt"))
    os.makedirs(os.path.join(code, "efficiency"))
    for f in ("pipeline.py", "fast_dicom.py", "fast_consensus.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(code, "efficiency"))
    import dynamic_network_architectures as dna
    shutil.copytree(os.path.dirname(dna.__file__), os.path.join(code, "dynamic_network_architectures"),
                    ignore=shutil.ignore_patterns("__pycache__"))

    w = os.path.join(a.out, "weights")
    shutil.rmtree(w, ignore_errors=True)
    for d in ("side", "roi"):
        half_copy(os.path.join(SUBMITTED, d), os.path.join(w, d))
    for m in sorted({m for v in PLUS.values() for m in v}):
        src = a.jc1 if m == "jc1" else EXTRA_SRC.get(m, os.path.join(SUBMITTED, "crops", m))
        half_copy(src, os.path.join(w, "crops", m))
    json.dump(RECIPE, open(os.path.join(w, "recipe.json"), "w"), indent=1)
    json.dump(PLUS, open(os.path.join(w, "recipe_plus.json"), "w"), indent=1)
    for d, slug, title in ((code, "rsna-knee-code-eff", "RSNA knee code (efficiency)"),
                           (w, "rsna-knee-weights-eff", "RSNA knee weights (efficiency)")):
        json.dump({"title": title, "id": f"theoviel/{slug}", "licenses": [{"name": "CC0-1.0"}]},
                  open(os.path.join(d, "dataset-metadata.json"), "w"), indent=1)
    for d in (code, w):
        n = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if os.path.isfile(f))
        print(f"{d}: {n / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
