"""Training crops for the pseudo-labelled fastMRI clinical DICOM studies, run from src/.

    python -m fastmri.dicom_crops --batch b2

Inputs: an inference run on the studies (output/fastmri_dicom/pl_<batch>: per-shard series.csv with the side and
conv_flip of every series, rois.csv with the consensus boxes, axial ones registered) and the staged prep of the same
series (output/fastmri_dicom/prep_<batch>: series.csv, cache/<SeriesInstanceUID>.npy, stored orientation).
`crops.prepare` flips a crop when its side is "L"; the caches are in stored orientation, so the side passed is the
effective one, as at inference: (side == "L") xor conv_flip.
Builds output/crops/fmd_<batch>_s1n8 (sagittal + coronal, 8 ROIs), fmd_<batch>_s1n8_axial (axial) and
fmd9_<batch>_s1n8 (+ the fibular head, fracture.fibula), all with the f_s1n8 sampling.
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from params import INPUT_FOLDER, OUTPUT_FOLDER

D = os.path.join(OUTPUT_FOLDER, "fastmri_dicom")
FLUID = ("fs", "fat", "t2", "stir", "spir")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", default="b2")
    a = p.parse_args()
    run, prep = os.path.join(D, f"pl_{a.batch}"), os.path.join(D, f"prep_{a.batch}")
    ser = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(run, "shard*", "series.csv"))])
    ser = ser[ser.status == "ok"]
    boxes = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(run, "shard*", "rois.csv"))])
    pr = pd.read_csv(os.path.join(prep, "series.csv"))
    pr = pr[pr.status == "ok"].set_index("SeriesInstanceUID")
    ser = ser[ser.SeriesInstanceUID.isin(pr.index)]
    # The prep and the pipeline must agree on the slices the boxes refer to.
    same = (pr.loc[ser.SeriesInstanceUID, ["n_slices", "n_cache", "rows", "cols"]].values == ser[["n_slices", "n_cache", "rows", "cols"]].values).all(1)
    print(f"{len(ser)} series in both; metadata identical for {same.mean():.4f}")
    ser = ser[same]
    desc = pd.read_csv(os.path.join(INPUT_FOLDER, "fastmri", "dicom_series.csv")).set_index("SeriesInstanceUID").description
    ser["Fluid_Sensitive"] = ser.SeriesInstanceUID.map(desc).fillna("").str.lower().apply(lambda s: int(any(k in s for k in FLUID)))
    ser["Anatomical_Plane"] = ser.plane
    eff = np.where((ser.side == "L") ^ ser.conv_flip.astype(bool), "L", "R")
    ser.assign(side=eff)[["SeriesInstanceUID", "side"]].to_csv(os.path.join(D, f"sides_{a.batch}.csv"), index=False)
    cols = ["SeriesInstanceUID", "StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "n_slices", "n_cache", "rows", "cols"]
    ser[ser.plane.isin(["Sagittal", "Coronal"])][cols].to_csv(os.path.join(D, f"volumes_{a.batch}.csv"), index=False)
    ser[ser.plane == "Axial"][cols].to_csv(os.path.join(D, f"volumes_axial_{a.batch}.csv"), index=False)
    boxes[boxes.SeriesInstanceUID.isin(ser.SeriesInstanceUID)].to_csv(os.path.join(D, f"rois_{a.batch}.csv"), index=False)
    py, cache = sys.executable, os.path.join(prep, "cache")
    common = ["--cache_dir", cache, "--side_csv", os.path.join(D, f"sides_{a.batch}.csv"), "--boxes", os.path.join(D, f"rois_{a.batch}.csv"),
              "--depth", "8", "--z_mode", "strided", "--stride", "1"]
    for name, vols in [(f"fmd_{a.batch}_s1n8", f"volumes_{a.batch}.csv"), (f"fmd_{a.batch}_s1n8_axial", f"volumes_axial_{a.batch}.csv")]:
        subprocess.run([py, "-m", "crops.prepare", "--name", name, "--volumes", os.path.join(D, vols)] + common, check=True)
    subprocess.run([py, "-m", "fracture.fibula", "--build", "--base", f"fmd_{a.batch}_s1n8", "--name", f"fmd9_{a.batch}_s1n8",
                    "--boxes", os.path.join(D, f"rois_{a.batch}.csv"), "--cache_dir", cache], check=True)


if __name__ == "__main__":
    main()
