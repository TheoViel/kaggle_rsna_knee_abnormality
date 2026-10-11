"""ROI boxes for SKM-TEA / MRNet series (after `skm.prep` / `mrnet.prep`), run from src/.

    python -m ext.boxes --data skm
    python -m ext.boxes --data mrnet

Side classifier + ROI U-Net as at test time (`infer.side`, `infer.seg.predict_boxes`), without the
cross-series consensus: SKM-TEA has one series per knee and MRNet has no geometry, so there is nothing
to register from. Every U-Net box is kept; confidence = the U-Net's mean probability. Writes into
output/<data>/: sides.csv, volumes.csv (crops.prepare's metadata format), rois.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

from params import OUTPUT_FOLDER

SIDE_W = "../logs/side_run/2026-10-09/0"
ROI_W = "../logs/roi_run/2026-10-09/1"


def main():
    from infer.seg import predict_boxes
    from infer.side import predict_sides

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, choices=["skm", "mrnet"])
    a = p.parse_args()
    d = os.path.join(OUTPUT_FOLDER, a.data)
    s = pd.read_csv(os.path.join(d, "series.csv"))
    s = s[s.status == "ok"].reset_index(drop=True)
    vols = np.load(os.path.join(d, "volumes.npy"), mmap_mode="r")
    pr = predict_sides(vols[s.row.to_numpy()], s.plane.tolist(), SIDE_W)
    s["p_right"] = pr
    if a.data == "mrnet":
        # One knee per exam: the planes vote (mean probability), so all series of an exam share a side.
        s["p_right"] = s.groupby("StudyInstanceUID").p_right.transform("mean")
        agree = (pd.Series(pr > 0.5) == (s.p_right > 0.5)).mean()
        print(f"series side agreeing with the exam vote: {agree:.3f}")
    s["side"] = np.where(s.p_right > 0.5, "R", "L")
    s[["SeriesInstanceUID", "side"]].to_csv(os.path.join(d, "sides.csv"), index=False)
    s.rename(columns={"plane": "Anatomical_Plane"})[
        ["SeriesInstanceUID", "StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "n_slices", "n_cache", "rows", "cols"]
    ].to_csv(os.path.join(d, "volumes.csv"), index=False)
    print("sides:", s.groupby("plane").side.value_counts().unstack().to_dict("index"))

    rows = s[s.plane.isin(["Sagittal", "Coronal"])].reset_index(drop=True)
    pred = predict_boxes(vols, rows, ROI_W)
    pred = pred[pred.z0.notna()].assign(confidence=lambda x: x.p_mean, source="unet")
    pred.to_csv(os.path.join(d, "rois.csv"), index=False)
    n = pred.groupby("SeriesInstanceUID").size()
    print(f"-> rois.csv: {len(pred)} boxes, {pred.SeriesInstanceUID.nunique()}/{len(rows)} series, "
          f"{(n == 8).mean():.3f} with all 8 classes")
    print(pred.groupby(["plane", "class"]).confidence.mean().unstack(0).round(3).to_string())


if __name__ == "__main__":
    main()
