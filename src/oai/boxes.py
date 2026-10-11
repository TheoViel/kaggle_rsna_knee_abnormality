"""ROI boxes for the OAI series (after `oai.prep`), run from src/.

    python -m oai.boxes

The test-time path (`infer.side`, `infer.seg`, `infer.classify.axial_boxes`) with one difference:
the knee side is known (series names say RIGHT / LEFT), so it is used directly; the side
classifier is run only as a check. Writes output/oai/rois.csv (sagittal / coronal, U-Net +
registration fill) and rois_axial.csv (registered onto the axial reformats), and side_check.csv.
"""

import json
import os

import numpy as np
import pandas as pd

import knee_registration as kr
from params import OUTPUT_FOLDER

OAI_DIR = os.path.join(OUTPUT_FOLDER, "oai")
SIDE_W = "../logs/side_run/2026-10-09/0"
ROI_W = "../logs/roi_run/2026-10-09/1"


def main():
    from infer.classify import axial_boxes
    from infer.seg import consensus, predict_boxes
    from infer.side import predict_sides

    s = pd.read_csv(os.path.join(OAI_DIR, "series.csv"))
    s = s[s.status == "ok"].reset_index(drop=True)
    with open(os.path.join(OAI_DIR, "geometry.json")) as f:
        kr.GEOMETRY_OVERRIDE.update(json.load(f))
    vols = np.load(os.path.join(OAI_DIR, "volumes.npy"), mmap_mode="r")
    # crops.prepare inputs: known sides; sagittal / coronal and axial (AX_MPR) series metadata.
    s[["SeriesInstanceUID", "side"]].to_csv(os.path.join(OAI_DIR, "sides.csv"), index=False)
    meta = s.rename(columns={"plane": "Anatomical_Plane"})[
        ["SeriesInstanceUID", "StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "n_slices", "n_cache", "rows", "cols", "seq"]]
    meta[meta.Anatomical_Plane.isin(["Sagittal", "Coronal"])].to_csv(os.path.join(OAI_DIR, "volumes.csv"), index=False)
    meta[meta.Anatomical_Plane.eq("Axial")].to_csv(os.path.join(OAI_DIR, "volumes_axial.csv"), index=False)

    p = predict_sides(vols[s.row.to_numpy()], s.plane.tolist(), SIDE_W)
    chk = s[["SeriesInstanceUID", "seq", "side"]].assign(p_right=p, side_model=np.where(p > 0.5, "R", "L"))
    chk.to_csv(os.path.join(OAI_DIR, "side_check.csv"), index=False)
    print("side classifier agrees with the series name:", chk.groupby("seq").apply(lambda g: (g.side == g.side_model).mean()).round(3).to_dict())

    rows = s[s.plane.isin(["Sagittal", "Coronal"])].reset_index(drop=True)
    pred = predict_boxes(vols, rows, ROI_W)
    rois, n_weak, n_filled = consensus(pred, rows)
    rois.to_csv(os.path.join(OAI_DIR, "rois.csv"), index=False)
    print(f"-> rois.csv: {len(rois)} boxes, {rois.SeriesInstanceUID.nunique()} series; {n_weak} weak, {n_filled} filled by registration; "
          f"mean confidence {rois.confidence.mean():.3f}")
    print(rois.merge(s[["SeriesInstanceUID", "seq"]]).groupby("seq").confidence.mean().round(3).to_dict())
    side = dict(zip(s.SeriesInstanceUID, s.side))
    ax = axial_boxes(s.assign(status="ok"), rois, side)
    ax.to_csv(os.path.join(OAI_DIR, "rois_axial.csv"), index=False)
    print(f"-> rois_axial.csv: {len(ax)} boxes, {ax.SeriesInstanceUID.nunique()} axial series")


if __name__ == "__main__":
    main()
