"""Pseudo-labels for the fastMRI clinical DICOM studies from an inference run of a weight folder, run from src/.

    python -m fastmri.pseudo_labels --run ../output/fastmri_dicom/pl_b2 --weights ../output/dataset_2_moaks_mrnet_men_frac

The run (infer.pipeline on input/fastmri/dicom_flat) writes per-shard preds.csv: one raw probability per (model,
study, finding), including plane-restricted entries ("name@Sagittal,Coronal") and per-ROI sub-targets
("<ROI>|fracture"). Per finding, the pseudo-label is the mean probability of the recipe's models for it
(probabilities, not the recipe's test-set ranks). Per-ROI fracture pseudo-labels: the mean over the fracture models
of their "<ROI>|fracture" outputs. Writes <run>/pl_study.csv (StudyInstanceUID x findings) and pl_roi.csv.
"""

import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--weights", required=True)
    a = p.parse_args()
    recipe = json.load(open(os.path.join(a.weights, "recipe.json")))
    preds = pd.concat([pd.read_csv(f, dtype={"model": str}) for f in glob.glob(os.path.join(a.run, "shard*", "preds.csv"))])
    tab = preds.pivot_table(index="StudyInstanceUID", columns=["model", "cls"], values="p")
    study = pd.DataFrame({c: tab[[(m, c) for m in (v if isinstance(v, list) else v["models"]) if (m, c) in tab]].mean(axis=1)
                          for c, v in recipe.items()})
    study.rename_axis("StudyInstanceUID").to_csv(os.path.join(a.run, "pl_study.csv"))
    frac = recipe.get("Fracture", [])
    roi_cols = sorted({c for m, c in tab.columns if m in frac and "|" in c})
    roi = pd.DataFrame({c: tab[[(m, c) for m in frac if (m, c) in tab]].mean(axis=1) for c in roi_cols})
    roi.rename_axis("StudyInstanceUID").to_csv(os.path.join(a.run, "pl_roi.csv"))
    print(f"{len(study)} studies; mean pseudo-label per finding:", study.mean().round(3).to_dict())
    print("share >= 0.5:", (study >= 0.5).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
