"""MCL boxes from anatomy, to bootstrap the one ROI the attention cannot give.

    python -m roi.mcl_prior --pred round1_pred --out mcl_prior

The classifier never attends to the MCL itself (its compact MCL maps sit in the
tibia, on the bone-marrow signal that goes with MCL injuries), so its ROI is
placed from the medial compartment, which the U-Net localises well:

  - per study, the medial compartment (Medial OA) boxes of the U-Net, in patient
    space: left-right and cranio-caudal extent from the coronal series (precise
    along those axes), antero-posterior from the sagittal ones (likewise), medians
    over series; one plane stands in for the other when missing;
  - medial is +x (patient left) for a right knee, -x for a left one
    (`roi.laterality.study_side`: the side classifier's majority over the study);
  - the superficial MCL runs along the outer medial edge of the knee, from the
    femoral epicondyle ~3-4 cm above the joint line to the tibia ~4-6 cm below,
    1-1.5 cm wide antero-posteriorly, over the middle of the compartment. The box:
      left-right   [edge - INSET, edge + DEPTH] outward from the compartment's medial edge
      antero-post. compartment centre +- HALF_WIDTH
      cranio-caud. joint line (compartment centre) - BELOW .. + ABOVE
    mapped into every sagittal and coronal series of the study.

These are labels from a prior, not from the image: the U-Net and the cross-view
rounds are what fit them to each knee.
"""

import argparse
import os

import numpy as np
import pandas as pd

from roi.boxes import _geometry, world_box
from roi.labels import BOX_COLS, LABEL_DIR
from roi.laterality import study_side

INSET, DEPTH = 3.0, 12.0          # mm, left-right around the compartment's medial edge
HALF_WIDTH = 10.0                 # mm, antero-posterior half width
BELOW, ABOVE = 45.0, 35.0         # mm, below / above the joint line


def world_to_voxel_box(series_uid, lo, hi, shape):
    """Patient-space box -> inclusive voxel box of a series (axis-aligned around the corners)."""
    origin, m = _geometry(series_uid)
    inv = np.linalg.inv(m)
    corners = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    idx = (corners - origin) @ inv.T                       # (x=col, y=row, z=slice) continuous
    a, b = idx.min(0), idx.max(0)
    n_slices, rows, cols = shape
    box = dict(x0=int(np.ceil(a[0] - 0.5)), x1=int(np.floor(b[0] + 0.5)),
               y0=int(np.ceil(a[1] - 0.5)), y1=int(np.floor(b[1] + 0.5)),
               z0=int(np.ceil(a[2] - 0.5)), z1=int(np.floor(b[2] + 0.5)))
    for k, n in (("x", cols), ("y", rows), ("z", n_slices)):
        if box[f"{k}1"] < 0 or box[f"{k}0"] > n - 1:
            return None                                    # outside the field of view
        box[f"{k}0"], box[f"{k}1"] = max(box[f"{k}0"], 0), min(box[f"{k}1"], n - 1)
    return box


def mcl_boxes(pred, series):
    side = study_side()
    comp = pred[(pred["class"] == "Medial OA") & (pred.conf >= 0.5)]
    out = []
    for study, g in comp.groupby("StudyInstanceUID"):
        if study not in side:
            continue
        w = {p: [world_box(r.SeriesInstanceUID, int(r.z0), int(r.z1), int(r.y0), int(r.y1), int(r.x0), int(r.x1))
                 for r in g[g.plane == p].itertuples()] for p in ("Coronal", "Sagittal")}
        cor = w["Coronal"] or w["Sagittal"]
        sag = w["Sagittal"] or w["Coronal"]
        lo_c, hi_c = np.median([b[0] for b in cor], 0), np.median([b[1] for b in cor], 0)
        lo_s, hi_s = np.median([b[0] for b in sag], 0), np.median([b[1] for b in sag], 0)
        med = 1.0 if side[study] == "R" else -1.0
        edge = hi_c[0] if med > 0 else lo_c[0]
        x = sorted([edge - med * INSET, edge + med * DEPTH])
        yc = (lo_s[1] + hi_s[1]) / 2
        zc = (lo_c[2] + hi_c[2]) / 2                       # LPS: +z is cranial
        lo = np.array([x[0], yc - HALF_WIDTH, zc - BELOW])
        hi = np.array([x[1], yc + HALF_WIDTH, zc + ABOVE])
        for r in series[series.StudyInstanceUID == study].itertuples():
            box = world_to_voxel_box(r.SeriesInstanceUID, lo, hi, (r.n_slices, r.rows, r.cols))
            if box is not None:
                out.append(dict(SeriesInstanceUID=r.SeriesInstanceUID, StudyInstanceUID=study,
                                plane=r.Anatomical_Plane, **{"class": "MCL"}, source="prior",
                                score=0.5, **box))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", default="round1_pred")
    p.add_argument("--out", default="mcl_prior")
    a = p.parse_args()
    pred = pd.read_csv(os.path.join(LABEL_DIR, f"{a.pred}.csv"))
    series = pd.read_csv("../output/roi/volumes.csv")
    boxes = mcl_boxes(pred, series)
    cols = ["SeriesInstanceUID", "StudyInstanceUID", "plane", "class", "source", "score"] + BOX_COLS
    boxes[cols].to_csv(os.path.join(LABEL_DIR, f"{a.out}.csv"), index=False)
    print(f"-> {len(boxes)} MCL boxes on {boxes.StudyInstanceUID.nunique()} studies "
          f"({boxes.plane.value_counts().to_dict()}), {os.path.join(LABEL_DIR, a.out + '.csv')}")


if __name__ == "__main__":
    main()
