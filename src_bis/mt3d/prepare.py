"""Joint crops for the multitask 3D model, run from src_bis/.

    python -m mt3d.prepare --name u10_d32_s256

One crop per sagittal / coronal series: the union of its 8 ROI boxes (U-Net
predictions, `--boxes`), widened in-plane by `--expand` x its height / width on
each side, as a square of its larger side, resized to SIZE x SIZE; through-plane,
DEPTH positions spread evenly over the union's slices widened by one, linearly
interpolated (`crops.prepare.crop_box`, z_mode "resize"). Left knees are mirrored
to look like right ones, as everywhere else.

Alongside each crop, the 8 ROI boxes in the crop's frame, as fractions of each
axis in edge coordinates (voxel i of n spans [i / n, (i + 1) / n)), so the
segmentation targets can be drawn at any resolution (`mt3d.model.box_masks`).

Output: output/bis/mt3d/<name>.npy (N, DEPTH, SIZE, SIZE) uint8, <name>.csv (row
-> series, study, plane, fluid / fat flags, side) and <name>_boxes.npy (N, 8, 6)
float32 as (z0, z1, y0, y1, x0, x1).
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from crops.prepare import capsule_boxes, crop_box, crop_window
from params import CACHE_DIR, INPUT_FOLDER, OUTPUT_FOLDER
from roi.extract import ROI_CLASSES

MT_DIR = os.path.join(OUTPUT_FOLDER, "bis", "mt3d")
BOX = ["z0", "z1", "y0", "y1", "x0", "x1"]


def roi_fractions(box, union, n_c, cs, n_slices, rows, cols, depth, size, flip, sagittal):
    """An ROI box (original voxels) -> its edges in the union crop, as fractions of each axis."""
    zs, y0, x0, n = crop_window(n_c, cs, union, n_slices, rows, cols, depth, 0.0, 0, "resize")
    s = max(rows, cols)
    oy, ox = (s - rows) // 2, (s - cols) // 2
    f = cs / s
    zc = (lambda z: z * (n_c - 1) / max(n_slices - 1, 1)) if n_c != n_slices else (lambda z: z)
    za, zb = zs[0], zs[-1]
    step = (zb - za) / max(depth - 1, 1)
    # Cache slice coordinate c -> crop edge coordinate: crop position i sits at za + i * step.
    ez = [((zc(box["z0"] - 0.5) - za) / step + 0.5) / depth, ((zc(box["z1"] + 0.5) - za) / step + 0.5) / depth]
    ey = [((box["y0"] + oy) * f - y0) / n, ((box["y1"] + 1 + oy) * f - y0) / n]
    ex = [((box["x0"] + ox) * f - x0) / n, ((box["x1"] + 1 + ox) * f - x0) / n]
    if flip:
        if sagittal:
            ez = [1 - ez[1], 1 - ez[0]]
        else:
            ex = [1 - ex[1], 1 - ex[0]]
    return np.clip(np.array(ez + ey + ex, np.float32), 0, 1)


def _series(args):
    import torch

    torch.set_num_threads(1)
    r, union, rois, flip, depth, size = args
    sag = r["Anatomical_Plane"] == "Sagittal"
    cache = np.load(os.path.join(CACHE_DIR, f"{r['SeriesInstanceUID']}.npy"))
    n_c, cs = cache.shape[0], cache.shape[1]
    crop = crop_box(cache, union, r["n_slices"], r["rows"], r["cols"], flip, sag, depth=depth, size=size,
                    margin=0.0, min_side=0, z_mode="resize")
    fr = np.stack([roi_fractions(rois[c], union, n_c, cs, r["n_slices"], r["rows"], r["cols"], depth, size, flip, sag)
                   for c in ROI_CLASSES])
    return crop, fr


def main():
    from roi.labels import LABEL_DIR
    from roi.laterality import series_side

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxes", default="round3_pred")
    p.add_argument("--name", default="u10_d32_s256")
    p.add_argument("--expand", type=float, default=0.10)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    os.makedirs(MT_DIR, exist_ok=True)
    vols = pd.read_csv(os.path.join(OUTPUT_FOLDER, "roi", "volumes.csv"))
    if a.limit:
        vols = vols.iloc[:a.limit]
    fat = pd.read_csv(os.path.join(INPUT_FOLDER, "train_series.csv")).set_index("SeriesInstanceUID").Fat_Suppression
    pred = pd.read_csv(os.path.join(LABEL_DIR, f"{a.boxes}.csv"))
    pred = pred[pred.z0.notna()]
    shape = {r.SeriesInstanceUID: (r.rows, r.cols) for r in vols.itertuples()}
    unions = capsule_boxes(pred[pred.SeriesInstanceUID.isin(shape)], a.expand, None, shape)
    rois = {}
    for r in pred.to_dict("records"):
        rois.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in BOX}
    side = series_side()
    jobs = [(r, unions[r["SeriesInstanceUID"]]["Capsule"], rois[r["SeriesInstanceUID"]],
             side.get(r["SeriesInstanceUID"]) == "L", a.depth, a.size) for r in vols.to_dict("records")]
    arr = np.lib.format.open_memmap(os.path.join(MT_DIR, f"{a.name}.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(jobs), a.depth, a.size, a.size))
    boxes = np.zeros((len(jobs), len(ROI_CLASSES), 6), np.float32)
    with ProcessPoolExecutor(96) as ex:
        for i, (c, fr) in enumerate(ex.map(_series, jobs, chunksize=8)):
            arr[i], boxes[i] = c, fr
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    arr.flush()
    np.save(os.path.join(MT_DIR, f"{a.name}_boxes.npy"), boxes)
    vols.assign(row=np.arange(len(vols)), side=vols.SeriesInstanceUID.map(side),
                Fat_Suppression=vols.SeriesInstanceUID.map(fat).fillna(0).astype(int)).to_csv(
        os.path.join(MT_DIR, f"{a.name}.csv"), index=False)
    json.dump(vars(a), open(os.path.join(MT_DIR, f"{a.name}.json"), "w"), indent=2)
    print(f"-> {len(jobs)} crops {a.depth}x{a.size}x{a.size} in {MT_DIR}/{a.name}.npy")


if __name__ == "__main__":
    main()
