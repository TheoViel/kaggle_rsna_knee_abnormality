"""Compare attention-extraction configs by how localised their maps are and how
well the sagittal and coronal boxes agree, run from src/.

    python -m roi.eval_attention --tags plain4,tta4,ens --modes zrun,cc3d

For each (tag, box mode, max_area threshold): per class, coverage (share of
series x class maps localised), `both` (share of studies with a localised box in
both views, i.e. where agreement can be checked), `overlap` (share of those
sagittal x coronal pairs whose boxes intersect in patient space) and mean IoU.
A better extraction localises more maps without the views disagreeing more.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from roi.boxes import cross_view, grid_box, localisation, to_voxels
from roi.extract import ATTN_DIR, ROI_CLASSES


def series_planes():
    idx = pd.read_csv("../output/series_index.csv").set_index("SeriesInstanceUID")
    return idx[["StudyInstanceUID", "Anatomical_Plane"]]


def candidate_boxes(tag, mode, mass=0.5):
    """Every (series, class) candidate box of a tag, with its localisation metrics."""
    info = series_planes()
    rows = []
    for path in sorted(glob.glob(os.path.join(ATTN_DIR, tag, "maps", "*.npz"))):
        uid = os.path.basename(path)[:-4]
        d = np.load(path)
        grid = d["xy"].shape[-1]
        rows_, cols_ = int(d["rows"]), int(d["cols"])
        for k, c in enumerate(ROI_CLASSES):
            b = grid_box(d["xy"][k], d["z"][k], mode=mode, mass=mass)
            rows.append(dict(StudyInstanceUID=info.loc[uid, "StudyInstanceUID"],
                             SeriesInstanceUID=uid, plane=info.loc[uid, "Anatomical_Plane"],
                             **{"class": c}, purity=b["purity"],
                             **localisation(b, len(d["z"][k]), grid, rows_, cols_),
                             **to_voxels(b, d["slices"], rows_, cols_, grid)))
    return pd.DataFrame(rows)


def summarise(cand, max_area, max_z_frac):
    loc = cand[(cand.area <= max_area) & (cand.z_frac <= max_z_frac)]
    cv = cross_view(loc)
    n_studies = cand.StudyInstanceUID.nunique()
    out = {}
    for c in ROI_CLASSES:
        both = cv[cv.cls == c].StudyInstanceUID.nunique()
        pairs = cv[cv.cls == c]
        out[c] = dict(coverage=(loc["class"] == c).sum() / max((cand["class"] == c).sum(), 1),
                      both=both / n_studies,
                      overlap=pairs.overlap.mean() if len(pairs) else np.nan,
                      iou=pairs.iou.mean() if len(pairs) else np.nan)
    t = pd.DataFrame(out).T
    t.loc["ALL"] = [len(loc) / len(cand), t.both.mean(), cv.overlap.mean(), cv.iou.mean()]
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="plain4,tta4,ens")
    p.add_argument("--modes", default="zrun,cc3d")
    p.add_argument("--max_areas", default="0.02,0.03,0.05")
    p.add_argument("--max_z_frac", type=float, default=0.5)
    a = p.parse_args()
    pd.set_option("display.width", 200)
    for tag in a.tags.split(","):
        for mode in a.modes.split(","):
            cand = candidate_boxes(tag, mode)
            for ma in map(float, a.max_areas.split(",")):
                t = summarise(cand, ma, a.max_z_frac)
                print(f"\n== {tag}  {mode}  max_area {ma}")
                print(t.round(3).to_string())


if __name__ == "__main__":
    main()
