"""Axial ROI boxes without geometry (MRNet), from the sagittal / coronal U-Net boxes, run from src/.

    python -m ext.axial_boxes --data mrnet [--cv]

Our axial boxes come from registration (DICOM geometry), which MRNet lacks. Instead, per ROI class, a
ridge regression maps the in-plane coordinates of the sagittal and coronal boxes (rows / columns,
normalised by the image size; slice indices are left out as they need geometry to interpret) to the
normalised axial box. It is fitted on our studies: U-Net sagittal / coronal boxes (round3_pred) ->
registered axial boxes (round3_axial). 5-fold CV by study (--cv): median 3D IoU 0.44-0.53 per class (a
fixed mean box: 0-0.35); 82-95% of boxes above IoU 0.3.

Assumes the external data stores its series like ours (same in-plane display convention, axial slices in
the same superior-inferior order): check the crops. Writes output/<data>/rois_axial.csv, volumes_axial.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from params import OUTPUT_FOLDER

B = ["z0", "z1", "y0", "y1", "x0", "x1"]
FEATS = ["s_y0", "s_y1", "s_x0", "s_x1", "c_y0", "c_y1", "c_x0", "c_x1"]
ROI = os.path.join(OUTPUT_FOLDER, "roi")


def normalise(d, dims):
    d = d.join(dims[["n_slices", "rows", "cols"]], on="SeriesInstanceUID")
    for a, n in [("z", "n_slices"), ("y", "rows"), ("x", "cols")]:
        d[a + "0"] = d[a + "0"] / d[n]
        d[a + "1"] = (d[a + "1"] + 1) / d[n]
    return d


def features(boxes, dims):
    p = normalise(boxes, dims)
    sag = p[p.plane == "Sagittal"].groupby(["StudyInstanceUID", "class"])[B].median().add_prefix("s_")
    cor = p[p.plane == "Coronal"].groupby(["StudyInstanceUID", "class"])[B].median().add_prefix("c_")
    return sag.join(cor, how="inner").dropna()


def iou3(a, b):
    inter = np.prod(np.clip(np.minimum(a[:, 1::2], b[:, 1::2]) - np.maximum(a[:, 0::2], b[:, 0::2]), 0, None), 1)
    va, vb = np.prod(a[:, 1::2] - a[:, 0::2], 1), np.prod(b[:, 1::2] - b[:, 0::2], 1)
    return inter / (va + vb - inter)


def main():
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="mrnet")
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()
    dims = pd.concat([pd.read_csv(os.path.join(ROI, "volumes.csv")), pd.read_csv(os.path.join(ROI, "volumes_axial.csv"))]
                     ).drop_duplicates("SeriesInstanceUID").set_index("SeriesInstanceUID")
    x = features(pd.read_csv(os.path.join(ROI, "labels", "round3_pred.csv")), dims)
    y = normalise(pd.read_csv(os.path.join(ROI, "labels", "round3_axial.csv")), dims).groupby(["StudyInstanceUID", "class"])[B].median()
    d = x.join(y, how="inner").dropna().reset_index()
    models = {}
    for c, g in d.groupby("class"):
        if a.cv:
            pr = np.zeros((len(g), 6))
            for tr, te in GroupKFold(5).split(g, groups=g.StudyInstanceUID):
                pr[te] = Ridge(1e-3).fit(g[FEATS].values[tr], g[B].values[tr]).predict(g[FEATS].values[te])
            print(f"{c:17s} CV median IoU {np.median(iou3(pr, g[B].values)):.3f}")
        models[c] = Ridge(1e-3).fit(g[FEATS].values, g[B].values)

    out_dir = os.path.join(OUTPUT_FOLDER, a.data)
    vols = pd.read_csv(os.path.join(out_dir, "volumes.csv"))
    ext_dims = vols.set_index("SeriesInstanceUID")
    xe = features(pd.read_csv(os.path.join(out_dir, "rois.csv")), ext_dims).reset_index()
    axial = vols[vols.Anatomical_Plane == "Axial"]
    axial.to_csv(os.path.join(out_dir, "volumes_axial.csv"), index=False)
    rows = []
    for r in axial.itertuples():
        for c, m in models.items():
            f = xe[(xe.StudyInstanceUID == r.StudyInstanceUID) & (xe["class"] == c)]
            if f.empty:
                continue
            z0, z1, y0, y1, x0, x1 = m.predict(f[FEATS].values)[0]
            box = dict(z0=z0 * r.n_slices, z1=z1 * r.n_slices - 1, y0=y0 * r.rows, y1=y1 * r.rows - 1, x0=x0 * r.cols, x1=x1 * r.cols - 1)
            box = {k: int(round(np.clip(v, 0, (r.n_slices if k[0] == "z" else r.rows if k[0] == "y" else r.cols) - 1)))
                   for k, v in box.items()}
            rows.append(dict(StudyInstanceUID=r.StudyInstanceUID, SeriesInstanceUID=r.SeriesInstanceUID, plane="Axial",
                             **{"class": c}, **box))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "rois_axial.csv"), index=False)
    print(f"-> {len(out)} axial boxes for {out.SeriesInstanceUID.nunique()} / {len(axial)} axial series")


if __name__ == "__main__":
    main()
