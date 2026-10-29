"""ROI crops for the finding classifiers, run from src/.

    python -m crops.prepare --boxes round3_pred --name r3
    python -m crops.prepare --boxes round3_axial --name r3_axial --volumes volumes_axial
    python -m crops.prepare --boxes ../input/rois.csv.gz --name r3   # the repository's boxes (round 3, all planes)

For every sagittal and coronal series and every ROI (`roi.extract.ROI_CLASSES`),
from the series' 384 px uint8 cache (`prepare_cache.py`):
  - in-plane: a square around the box centre, side = max(box h, box w) x
    (1 + 2 x MARGIN), at least MIN_SIDE cache pixels, resized to SIZE x SIZE --
    a square keeps the anatomy's aspect ratio, the margin gives context;
  - through-plane: DEPTH slices spread evenly over the box's slice range widened
    by one slice each side, nearest slice (no interpolation between slices: a
    2.5D model reads real slices);
  - canonical orientation: left knees (side classifier, `roi.laterality.series_side`)
    are mirrored along the left-right axis -- the slice order for sagittal, the
    columns for coronal -- so every crop is seen as a right knee.

Boxes come from the ROI U-Net's predictions (`--boxes`, output/roi/labels/<name>.csv, or a file path),
which is also what inference produces, so training and test crops match.

Output: output/crops/<name>.npy (N, K, DEPTH, SIZE, SIZE) uint8, memory-mappable,
and <name>.csv (row -> series, plane, study).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import CACHE_DIR, OUTPUT_FOLDER
from roi.extract import ROI_CLASSES

CROP_DIR = os.path.join(OUTPUT_FOLDER, "crops")
DEPTH, SIZE = 12, 128
MARGIN, MIN_SIDE = 0.25, 48
BOX = ["z0", "z1", "y0", "y1", "x0", "x1"]


def crop_window(n_c, cs, box, n_slices, rows, cols, depth=DEPTH, margin=MARGIN, min_side=MIN_SIDE):
    """Where `crop_box` reads in a (n_c, cs, cs) cache: slice indices, window origin, window side."""
    s = max(rows, cols)
    oy, ox = (s - rows) // 2, (s - cols) // 2
    f = cs / s
    # In-plane: box centre and side in cache pixels.
    cy = ((box["y0"] + box["y1"] + 1) / 2 + oy) * f
    cx = ((box["x0"] + box["x1"] + 1) / 2 + ox) * f
    side = max((box["y1"] - box["y0"] + 1) * f, (box["x1"] - box["x0"] + 1) * f) * (1 + 2 * margin)
    side = max(side, min_side)
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    n = int(round(side))
    # Slices: box range widened by one, mapped to cache indices.
    zc = (lambda z: z * (n_c - 1) / max(n_slices - 1, 1)) if n_c != n_slices else (lambda z: z)
    za, zb = zc(max(box["z0"] - 1, 0)), zc(min(box["z1"] + 1, n_slices - 1))
    zs = np.clip(np.round(np.linspace(za, zb, depth)).astype(int), 0, n_c - 1)
    return zs, y0, x0, n


def crop_box(cache, box, n_slices, rows, cols, flip, sagittal, depth=DEPTH, size=SIZE,
             margin=MARGIN, min_side=MIN_SIDE):
    """One (depth, size, size) uint8 crop of a (n_c, 384, 384) cache, see module doc."""
    import torch
    import torch.nn.functional as F

    n_c, cs = cache.shape[0], cache.shape[1]
    zs, y0, x0, n = crop_window(n_c, cs, box, n_slices, rows, cols, depth, margin, min_side)
    # Zero-padded window so crops reaching past the edge keep their geometry.
    win = np.zeros((depth, n, n), np.float32)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + n, cs), min(x0 + n, cs)
    if ye > ys and xe > xs:
        win[:, ys - y0:ye - y0, xs - x0:xe - x0] = cache[zs][:, ys:ye, xs:xe]
    t = torch.from_numpy(win)[:, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", antialias=True, align_corners=False)[:, 0]
    out = t.round().clamp(0, 255).to(torch.uint8).numpy()
    if flip:
        out = out[::-1] if sagittal else out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _series(args):
    import torch

    torch.set_num_threads(1)
    r, boxes, flip = args
    cache = np.load(os.path.join(CACHE_DIR, f"{r['SeriesInstanceUID']}.npy"))
    out = np.zeros((len(ROI_CLASSES), DEPTH, SIZE, SIZE), np.uint8)
    for k, c in enumerate(ROI_CLASSES):
        if c in boxes:
            out[k] = crop_box(cache, boxes[c], r["n_slices"], r["rows"], r["cols"], flip,
                              r["Anatomical_Plane"] == "Sagittal")
    return out


def main():
    from roi.labels import LABEL_DIR
    from roi.laterality import series_side

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxes", default="round3_pred")
    p.add_argument("--name", default="r3")
    p.add_argument("--volumes", default="volumes", help="output/roi/<volumes>.csv: volumes (sag + cor) or volumes_axial")
    a = p.parse_args()
    os.makedirs(CROP_DIR, exist_ok=True)
    vols = pd.read_csv(os.path.join(OUTPUT_FOLDER, "roi", f"{a.volumes}.csv"))
    pred = pd.read_csv(a.boxes if os.path.isfile(a.boxes) else os.path.join(LABEL_DIR, f"{a.boxes}.csv"))
    pred = pred[pred.z0.notna()]
    boxes = {}
    for r in pred.to_dict("records"):
        boxes.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in BOX}
    side = series_side()
    jobs = [(r, boxes.get(r["SeriesInstanceUID"], {}), side.get(r["SeriesInstanceUID"]) == "L")
            for r in vols.to_dict("records")]
    arr = np.lib.format.open_memmap(os.path.join(CROP_DIR, f"{a.name}.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(jobs), len(ROI_CLASSES), DEPTH, SIZE, SIZE))
    with ProcessPoolExecutor(96) as ex:
        for i, c in enumerate(ex.map(_series, jobs, chunksize=8)):
            arr[i] = c
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    arr.flush()
    vols.assign(row=np.arange(len(vols)), side=vols.SeriesInstanceUID.map(side)).to_csv(
        os.path.join(CROP_DIR, f"{a.name}.csv"), index=False)
    print(f"-> {len(jobs)} series x {len(ROI_CLASSES)} ROIs x {DEPTH} x {SIZE} x {SIZE} in {CROP_DIR}")


if __name__ == "__main__":
    main()
