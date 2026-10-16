"""Three more fracture ROIs: the whole patella, the tibial tuberosity / anterior tibia, and the fibular neck / shaft,
derived from the U-Net's ROI boxes, and their crops; run from src/.

    python -m fracture.extra --check --uids <uid,...> --out ../output/locations/extra_check.png
    python -m fracture.extra --build          # output/crops/f16_s1n8.{npy,csv,json}: f13_s1n8 + 3 crops

Why: out-of-fold, the 13-crop fracture model (fb) ranks patella fractures lowest of all locations (0.931 against
clear negatives, vs 0.97-0.98 for the tibia). The PF OA box covers the patellofemoral joint surfaces, not the poles
or the tendon insertions; in coronal series the patella sits in the most anterior slices, where no box reaches. The
fibular-head crop stops at the head ("fibula, level not specified": 0.928), and nothing covers the tibial
tuberosity.

Boxes, in each series' own orientation (read from its boxes, as fracture.fibula does):
  Patella             the PF OA box, rows widened by 50% of its height on each side, the in-plane width axis by
                      25%, slices by 30% of the box's slice span on each side (covers both poles and the tendon
                      insertions, and the anterior coronal slices)
  Tibial Tuberosity   left-right: from the medial to the lateral meniscus centre, widened by 50%; rows: from the
                      joint line (meniscus centre) to 1.2 S below it (S: the meniscus box's in-plane size, as in
                      fracture.bones); front-back: from the ACL box centre to the PF box's anterior edge + 10%
  Fibula Shaft        the fibula box (fracture.fibula) moved down by its own height and stretched to twice it
Crops: the base set's through-plane sampling (8 strided slices), 128 px, margin 0.25.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from crops.prepare import BOX, CROP_DIR, crop_box
from fracture.fibula import BOXES, _c, boxes_per_series, fibula_box
from params import CACHE_DIR

EXTRA = ["Patella", "Tibial Tuberosity", "Fibula Shaft"]
NEEDED = ("Medial Meniscus", "Lateral Meniscus", "PF OA", "ACL", "Baker's")


def extra_boxes(bx, plane, rows, cols, n_slices):
    """ROI boxes of one sagittal / coronal series -> {extra ROI: box} (missing when the needed boxes are)."""
    if not set(NEEDED) <= set(bx):
        return {}
    mm, lm, pf, acl, bk = (bx[c] for c in NEEDED)
    lr, ap = ("z", "x") if plane == "Sagittal" else ("x", "z")
    lim = {"z": n_slices - 1, "y": rows - 1, "x": cols - 1}
    clip = lambda b: {k: int(np.clip(round(b[k]), 0, lim[k[0]])) for k in BOX}
    out = {}
    # Patella: the PF box, widened
    grow = {"y": 0.5, "x": 0.25, "z": 0.3}
    out["Patella"] = clip({k: pf[k] + (-1 if k.endswith("0") else 1) * grow[k[0]] * (pf[k[0] + "1"] - pf[k[0] + "0"] + 1)
                           for k in BOX})
    ant = 1 if _c(pf, ap) > _c(bk, ap) else -1                 # anterior along the AP axis
    inf = 1 if _c(mm, "y") > _c(pf, "y") else -1               # inferior along rows
    # Tibial tuberosity / anterior tibia
    a, b = _c(mm, lr), _c(lm, lr)
    w = abs(b - a) + 1
    lo, hi = min(a, b) - 0.25 * w, max(a, b) + 0.25 * w
    S = mm["x1"] - mm["x0"] + 1
    joint = _c(mm, "y")
    y_a, y_b = joint, joint + inf * 1.2 * S
    front = pf[f"{ap}1"] if ant > 0 else pf[f"{ap}0"]
    d = pf[f"{ap}1"] - pf[f"{ap}0"] + 1
    ap_a, ap_b = _c(acl, ap), front + ant * 0.1 * d
    tt = {f"{lr}0": lo, f"{lr}1": hi, "y0": min(y_a, y_b), "y1": max(y_a, y_b), f"{ap}0": min(ap_a, ap_b), f"{ap}1": max(ap_a, ap_b)}
    out["Tibial Tuberosity"] = clip(tt)
    # Fibula shaft: the fibula box moved down by its height, twice as tall
    fb = fibula_box(bx, plane, rows, cols, n_slices)
    if fb is not None:
        h = fb["y1"] - fb["y0"] + 1
        top = fb["y1"] if inf > 0 else fb["y0"]
        y_a, y_b = top, top + inf * 2 * h
        out["Fibula Shaft"] = clip(dict(fb, y0=min(y_a, y_b), y1=max(y_a, y_b)))
    return out


def _crop(job):
    import torch

    torch.set_num_threads(1)
    r, boxes, cache_dir, kw = job
    out = np.zeros((len(EXTRA), kw["depth"], 128, 128), np.uint8)
    if not boxes:
        return out
    cache = np.load(os.path.join(cache_dir, f"{r['SeriesInstanceUID']}.npy"))
    for i, c in enumerate(EXTRA):
        if c in boxes:
            out[i] = crop_box(cache, boxes[c], r["n_slices"], r["rows"], r["cols"], r["side"] == "L",
                              r["Anatomical_Plane"] == "Sagittal", size=128, margin=0.25, **kw)
    return out


def jobs(info, bx, cache_dir=CACHE_DIR, kw=None):
    kw = kw or dict(depth=8, z_mode="strided", stride=1)
    return [(r, extra_boxes(bx.get(r["SeriesInstanceUID"], {}), r["Anatomical_Plane"], r["rows"], r["cols"], r["n_slices"]),
             cache_dir, kw) for r in info.to_dict("records")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--uids", default="")
    p.add_argument("--out", default="")
    p.add_argument("--base", default="f13_s1n8")
    p.add_argument("--name", default="f16_s1n8")
    a = p.parse_args()
    info = pd.read_csv(os.path.join(CROP_DIR, f"{a.base}.csv"))
    meta = json.load(open(os.path.join(CROP_DIR, f"{a.base}.json")))
    kw = {k: meta[k] for k in ("depth", "z_mode", "stride") if k in meta}
    bx = boxes_per_series(BOXES)
    if a.check:
        from PIL import Image, ImageDraw

        sel = info[info.StudyInstanceUID.isin(a.uids.split(","))].groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(1)
        strips = []
        for j in jobs(sel, bx, kw=kw):
            c = _crop(j)
            img = Image.fromarray(np.concatenate([np.concatenate(list(c[k, ::2]), 1) for k in range(len(EXTRA))], 1)).convert("RGB")
            ImageDraw.Draw(img).text((3, 3), f"{j[0]['StudyInstanceUID'][-8:]} {j[0]['Anatomical_Plane']}  " + " | ".join(EXTRA), fill=(255, 255, 0))
            strips.append(np.array(img))
        Image.fromarray(np.concatenate(strips, 0)).save(a.out)
        print(f"{len(strips)} series -> {a.out}")
    if a.build:
        base = np.load(os.path.join(CROP_DIR, f"{a.base}.npy"), mmap_mode="r")
        n0 = base.shape[1]
        arr = np.lib.format.open_memmap(os.path.join(CROP_DIR, f"{a.name}.npy"), mode="w+", dtype=np.uint8,
                                        shape=(base.shape[0], n0 + len(EXTRA), *base.shape[2:]))
        js = jobs(info, bx, kw=kw)
        n_none = sum(len(b) < len(EXTRA) for _, b, _, _ in js)
        with ProcessPoolExecutor(96) as ex:
            for i, c in enumerate(ex.map(_crop, js, chunksize=16)):
                arr[i, :n0] = base[i]
                arr[i, n0:] = c
        arr.flush()
        info.to_csv(os.path.join(CROP_DIR, f"{a.name}.csv"), index=False)
        json.dump({**meta, "extra3": True}, open(os.path.join(CROP_DIR, f"{a.name}.json"), "w"))
        print(f"-> {a.name}: {arr.shape}; {n_none} series missing at least one extra box")


if __name__ == "__main__":
    main()
