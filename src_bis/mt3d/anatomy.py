"""Anatomy targets for the joint crops, from the nnU-Net knee segmentations, run from src_bis/.

    python -m mt3d.anatomy --crops u10_d32_s256

`input/segs_nnunet/<series>.npy` (`segment_nnunet.py`): uint8 on the original grid,
0 background, 1 patellar cartilage, 2 femoral cartilage, 3 medial tibial cartilage,
4 lateral tibial cartilage, 5 medial meniscus, 6 lateral meniscus, 7 femur, 8 tibia,
9 patella. About half the sagittal / coronal series have one.

The segmentation model works on fluid-sensitive series only: on the others the
cartilage and menisci are almost always missing and even the bones are patchy
(coronal T1 / PD). So only fluid-sensitive series get anatomy targets, and within
them a structure counts as labelled only when present (>= MIN_VOX voxels in the
crop); an absent structure is unknown, not background (missed by the model, or
outside the field of view).

Each label map is cut with the joint crop's own geometry (`mt3d.prepare`: union
box, square window, DEPTH positions over the union's slices, left knees
mirrored), nearest neighbour, at DEPTH x SIZE x SIZE (coarser than the image:
the model is supervised at 1/4 resolution and below).

Output: output/bis/mt3d/<crops>_anat.npy (N, DEPTH, SIZE, SIZE) uint8 and
<crops>_anat_lab.npy (N, 9) bool (labelled structures; all False without a map).
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from crops.prepare import capsule_boxes, crop_window
from mt3d.prepare import BOX, MT_DIR
from params import INPUT_FOLDER, OUTPUT_FOLDER

ANATOMY = ["patellar cartilage", "femoral cartilage", "medial tibial cartilage", "lateral tibial cartilage",
           "medial meniscus", "lateral meniscus", "femur", "tibia", "patella"]
SEG_DIR = os.path.join(INPUT_FOLDER, "segs_nnunet")
SIZE, MIN_VOX = 128, 20


def _series(args):
    r, union, flip, depth = args
    seg = np.load(os.path.join(SEG_DIR, f"{r['SeriesInstanceUID']}.npy"))
    n_s, rows, cols, n_c, cs = r["n_slices"], r["rows"], r["cols"], r["n_cache"], 384
    zs, y0, x0, n = crop_window(n_c, cs, union, n_s, rows, cols, depth, 0.0, 0, "resize")
    s = max(rows, cols)
    oy, ox = (s - rows) // 2, (s - cols) // 2
    f = cs / s
    # Crop voxel centres -> original voxels (nearest): cache slice -> original slice, cache px -> original px.
    zo = np.round(zs * (n_s - 1) / max(n_c - 1, 1) if n_c != n_s else zs).astype(int).clip(0, n_s - 1)
    yo = np.floor((y0 + (np.arange(SIZE) + 0.5) * n / SIZE) / f - oy).astype(int)
    xo = np.floor((x0 + (np.arange(SIZE) + 0.5) * n / SIZE) / f - ox).astype(int)
    vy, vx = (yo >= 0) & (yo < rows), (xo >= 0) & (xo < cols)
    out = seg[zo][:, yo.clip(0, rows - 1)][:, :, xo.clip(0, cols - 1)]
    out[:, ~vy] = 0
    out[:, :, ~vx] = 0
    if flip:
        out = out[::-1] if r["Anatomical_Plane"] == "Sagittal" else out[:, :, ::-1]
    lab = np.bincount(out.ravel(), minlength=10)[1:] >= MIN_VOX
    return np.ascontiguousarray(out), lab


def main():
    from roi.labels import LABEL_DIR

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", default="u10_d32_s256")
    a = p.parse_args()
    meta = json.load(open(os.path.join(MT_DIR, f"{a.crops}.json")))
    info = pd.read_csv(os.path.join(MT_DIR, f"{a.crops}.csv"))
    pred = pd.read_csv(os.path.join(LABEL_DIR, f"{meta['boxes']}.csv"))
    pred = pred[pred.z0.notna()]
    shape = {r.SeriesInstanceUID: (r.rows, r.cols) for r in info.itertuples()}
    unions = capsule_boxes(pred[pred.SeriesInstanceUID.isin(shape)], meta["expand"], None, shape)
    have = {f[:-4] for f in os.listdir(SEG_DIR)}
    use = info.SeriesInstanceUID.isin(have) & info.Fluid_Sensitive.eq(1)
    rows = info[use].to_dict("records")
    jobs = [(r, unions[r["SeriesInstanceUID"]]["Capsule"], r["side"] == "L", meta["depth"]) for r in rows]
    arr = np.lib.format.open_memmap(os.path.join(MT_DIR, f"{a.crops}_anat.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(info), meta["depth"], SIZE, SIZE))
    lab = np.zeros((len(info), len(ANATOMY)), bool)
    with ProcessPoolExecutor(96) as ex:
        for r, (m, l) in zip(rows, ex.map(_series, jobs, chunksize=4)):
            arr[r["row"]], lab[r["row"]] = m, l
    arr.flush()
    np.save(os.path.join(MT_DIR, f"{a.crops}_anat_lab.npy"), lab)
    print(f"-> {len(rows)} series with anatomy of {len(info)}; labelled share per structure:")
    print(pd.Series(lab[use.to_numpy()].mean(0), ANATOMY).round(3).to_string())


if __name__ == "__main__":
    main()
