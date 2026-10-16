"""Hoffa's (infrapatellar) fat pad crop for synovitis, derived from the ROI boxes of sagittal series; run from src/.

    python -m crops.hoffa --check --uids <uid,uid,...> --out ../output/locations/hoffa_check.png
    python -m crops.hoffa --build              # output/crops/hoffa_u12.{npy,csv,json}: one crop per sagittal series

Why: the OAI model whose pseudo-labels lift synovitis on the 58 (0.878) is trained on MOAKS *Hoffa*-synovitis, graded
in the infrapatellar fat pad; its effusion-synovitis head scores only 0.780. The capsule crop squeezes the whole joint
into 160 px, so the fat pad is a small part of it. This crop shows the fat pad at about ROI resolution.

Box, in the series' own orientation (read from its boxes, as fracture.fibula does):
  left-right (slices)  from the medial meniscus centre to the lateral meniscus centre (the fat pad spans the notch
                       and both compartments' front)
  rows                 from the patella box centre down to the joint line (meniscus centre) + 0.3 x meniscus height
  front-back (cols)    from the ACL box's anterior edge to the patella (PF) box's anterior edge
Crop: 12 frames spread over the slices (z_mode uniform), 160 px, margin 0.15.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from crops.prepare import BOX, CROP_DIR, crop_box
from fracture.fibula import BOXES, _c, boxes_per_series
from params import CACHE_DIR

NEEDED = ("Medial Meniscus", "Lateral Meniscus", "PF OA", "ACL", "Baker's")
KW = dict(depth=12, z_mode="uniform", size=160, margin=0.15)


def hoffa_box(bx, rows, cols, n_slices):
    """ROI boxes of one sagittal series -> the fat pad box, or None."""
    if not set(NEEDED) <= set(bx):
        return None
    mm, lm, pf, acl, bk = (bx[c] for c in NEEDED)
    ant = 1 if _c(pf, "x") > _c(bk, "x") else -1              # anterior along columns
    inf = 1 if _c(mm, "y") > _c(pf, "y") else -1               # inferior along rows
    lim = {"z": n_slices - 1, "y": rows - 1, "x": cols - 1}
    z_a, z_b = _c(mm, "z"), _c(lm, "z")
    h = mm["y1"] - mm["y0"] + 1
    y_a, y_b = _c(pf, "y"), _c(mm, "y") + inf * 0.3 * h
    acl_front = acl["x1"] if ant > 0 else acl["x0"]
    pf_front = pf["x1"] if ant > 0 else pf["x0"]
    out = dict(z0=min(z_a, z_b), z1=max(z_a, z_b), y0=min(y_a, y_b), y1=max(y_a, y_b),
               x0=min(acl_front, pf_front), x1=max(acl_front, pf_front))
    return {k: int(np.clip(round(out[k]), 0, lim[k[0]])) for k in BOX}


def _crop(job):
    import torch

    torch.set_num_threads(1)
    r, box, cache_dir = job
    if box is None:
        return np.zeros((KW["depth"], KW["size"], KW["size"]), np.uint8)
    cache = np.load(os.path.join(cache_dir, f"{r['SeriesInstanceUID']}.npy"))
    return crop_box(cache, box, r["n_slices"], r["rows"], r["cols"], r["side"] == "L", True, **KW)


def jobs(info, bx, cache_dir=CACHE_DIR):
    return [(r, hoffa_box(bx.get(r["SeriesInstanceUID"], {}), r["rows"], r["cols"], r["n_slices"]), cache_dir)
            for r in info.to_dict("records")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--uids", default="")
    p.add_argument("--out", default="")
    p.add_argument("--base", default="cap_e15_d24", help="crop set whose sagittal rows (series metadata) are used")
    p.add_argument("--name", default="hoffa_u12")
    a = p.parse_args()
    info = pd.read_csv(os.path.join(CROP_DIR, f"{a.base}.csv"))
    info = info[info.Anatomical_Plane == "Sagittal"].reset_index(drop=True)
    info["row"] = np.arange(len(info))                  # rows of this crop set (the base set's rows are not ours)
    bx = boxes_per_series(BOXES)
    if a.check:
        from PIL import Image, ImageDraw

        sel = info[info.StudyInstanceUID.isin(a.uids.split(","))].groupby("StudyInstanceUID").head(1)
        strips = []
        for j in jobs(sel, bx):
            c = _crop(j)
            img = Image.fromarray(np.concatenate(list(c[::2]), 1)).convert("RGB")
            ImageDraw.Draw(img).text((3, 3), f"{j[0]['StudyInstanceUID'][-8:]} box {j[1]}", fill=(255, 255, 0))
            strips.append(np.array(img))
        Image.fromarray(np.concatenate(strips, 0)).save(a.out)
        print(f"{len(strips)} series -> {a.out}")
    if a.build:
        arr = np.lib.format.open_memmap(os.path.join(CROP_DIR, f"{a.name}.npy"), mode="w+", dtype=np.uint8,
                                        shape=(len(info), 1, KW["depth"], KW["size"], KW["size"]))
        js = jobs(info, bx)
        n_none = sum(b is None for _, b, _ in js)
        with ProcessPoolExecutor(96) as ex:
            for i, c in enumerate(ex.map(_crop, js, chunksize=16)):
                arr[i, 0] = c
        arr.flush()
        info.to_csv(os.path.join(CROP_DIR, f"{a.name}.csv"), index=False)
        json.dump({**KW, "stride": 1, "hoffa": True}, open(os.path.join(CROP_DIR, f"{a.name}.json"), "w"))
        print(f"-> {a.name}: {arr.shape}; {n_none} series without a fat pad box (missing ROI boxes)")


if __name__ == "__main__":
    main()
