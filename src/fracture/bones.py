"""Four bone ROIs for fracture: medial / lateral distal femur and proximal tibia, derived from the U-Net's ROI
boxes, and their crops; run from src/.

    python -m fracture.bones --check --out ../output/locations/bones_check.png   # crops of a few fracture studies
    python -m fracture.bones --build              # output/crops/f13_s1n8.{npy,csv,json}: f9_s1n8 + 4 bone crops

Why: the OA and meniscus boxes are bands around the joint line (in coronal series especially), so most of the
femoral condyles above and the tibial metaphysis below them, where impaction fractures and their marrow edema sit,
lie outside every crop. Each bone box covers one compartment's bone, from the joint line into the shaft:
  left-right   the union of the compartment's meniscus and OA boxes, widened by 25% of its width on each side;
  front-back   the same union along the AP axis, widened by 15% on each side;
  rows         femur: from 0.15 S below the joint line (the meniscus box centre) to 1.2 S above it;
               tibia: from 0.15 S above it to 1.0 S below it;
               S = the in-plane, non-vertical size of the meniscus box (its width in coronal series, its AP length
               in sagittal series: ~3-4 cm), so that the box scales with the knee and not with the FOV.
Orientation (left-right, inferior, posterior) is read from the series' own boxes, as for the fibula
(fracture.fibula). Crops: the f_s1n8 sampling (8 strided slices, 128 px, margin 0.25), rows of f9_s1n8.
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

BONES = ["Medial Femur", "Lateral Femur", "Medial Tibia", "Lateral Tibia"]
NEEDED = ("Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA")


def bone_boxes(bx, plane, rows, cols, n_slices):
    """ROI boxes of one sagittal / coronal series ({class: box}) -> {bone: box} (empty if boxes are missing)."""
    if not set(NEEDED) <= set(bx):
        return {}
    lr, ap = ("z", "x") if plane == "Sagittal" else ("x", "z")
    inplane = "x"                                              # the in-plane axis that is not rows
    lim = {"z": n_slices - 1, "y": rows - 1, "x": cols - 1}
    inf = 1 if _c(bx["Medial Meniscus"], "y") > _c(bx["PF OA"], "y") else -1      # inferior along rows
    out = {}
    for side in ("Medial", "Lateral"):
        men, oa = bx[f"{side} Meniscus"], bx[f"{side} OA"]
        b = {}
        for a, grow in ((lr, 0.25), (ap, 0.15)):
            lo, hi = min(men[f"{a}0"], oa[f"{a}0"]), max(men[f"{a}1"], oa[f"{a}1"])
            w = hi - lo + 1
            b[f"{a}0"], b[f"{a}1"] = lo - grow * w, hi + grow * w
        S = men[f"{inplane}1"] - men[f"{inplane}0"] + 1
        joint = _c(men, "y")
        for bone, (near, far) in (("Femur", (-0.15, 1.2)), ("Tibia", (-0.15, 1.0))):
            d = -inf if bone == "Femur" else inf                  # direction away from the joint along rows
            y_a, y_b = joint - d * 0.15 * S, joint + d * far * S
            box = dict(b, y0=min(y_a, y_b), y1=max(y_a, y_b))
            out[f"{side} {bone}"] = {k: int(np.clip(round(box[k]), 0, lim[k[0]])) for k in BOX}
    return out


def _crop(job):
    import torch

    torch.set_num_threads(1)
    r, boxes, cache_dir, kw = job if len(job) == 4 else (*job, dict(depth=8, z_mode="strided", stride=1))
    if not boxes:
        return np.zeros((len(BONES), kw["depth"], 128, 128), np.uint8)
    cache = np.load(os.path.join(cache_dir, f"{r['SeriesInstanceUID']}.npy"))
    return np.stack([crop_box(cache, boxes[b], r["n_slices"], r["rows"], r["cols"], r["side"] == "L",
                              r["Anatomical_Plane"] == "Sagittal", size=128, margin=0.25, **kw) for b in BONES])


def jobs(info, bx, cache_dir=CACHE_DIR, kw=None):
    """kw: through-plane sampling (the base crop set's, see main); f_s1n8's by default."""
    kw = kw or dict(depth=8, z_mode="strided", stride=1)
    return [(r, bone_boxes(bx.get(r["SeriesInstanceUID"], {}), r["Anatomical_Plane"], r["rows"], r["cols"], r["n_slices"]),
             cache_dir, kw) for r in info.to_dict("records")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--uids", default="", help="--check: comma-separated StudyInstanceUIDs")
    p.add_argument("--base", default="f9_s1n8", help="9-ROI crop set to extend")
    p.add_argument("--name", default="f13_s1n8")
    p.add_argument("--boxes", default=BOXES)
    p.add_argument("--cache_dir", default=CACHE_DIR)
    a = p.parse_args()
    info = pd.read_csv(os.path.join(CROP_DIR, f"{a.base}.csv"))
    bx = boxes_per_series(a.boxes)
    if a.check:
        from PIL import Image, ImageDraw

        sel = info[info.StudyInstanceUID.isin(a.uids.split(","))].groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(1)
        strips = []
        for j in jobs(sel, bx, a.cache_dir):
            c = _crop(j)                                   # (4, 8, 128, 128): bones x slices
            img = Image.fromarray(np.concatenate([np.concatenate(list(c[k, ::2]), 1) for k in range(4)], 1)).convert("RGB")
            ImageDraw.Draw(img).text((3, 3), f"{j[0]['StudyInstanceUID'][-8:]} {j[0]['Anatomical_Plane']}  "
                                     + " | ".join(BONES), fill=(255, 255, 0))
            strips.append(np.array(img))
        Image.fromarray(np.concatenate(strips, 0)).save(a.out)
        print(f"{len(strips)} series -> {a.out}")
    if a.build:
        base = np.load(os.path.join(CROP_DIR, f"{a.base}.npy"), mmap_mode="r")
        n0 = base.shape[1]
        arr = np.lib.format.open_memmap(os.path.join(CROP_DIR, f"{a.name}.npy"), mode="w+", dtype=np.uint8,
                                        shape=(base.shape[0], n0 + len(BONES), *base.shape[2:]))
        meta = json.load(open(os.path.join(CROP_DIR, f"{a.base}.json")))
        js = jobs(info, bx, a.cache_dir, {k: meta[k] for k in ("depth", "z_mode", "stride") if k in meta})
        n_none = sum(not b for _, b, _, _ in js)
        with ProcessPoolExecutor(96) as ex:
            for i, c in enumerate(ex.map(_crop, js, chunksize=16)):
                arr[i, :n0] = base[i]
                arr[i, n0:] = c
        arr.flush()
        info.to_csv(os.path.join(CROP_DIR, f"{a.name}.csv"), index=False)
        json.dump({**meta, "bones": True}, open(os.path.join(CROP_DIR, f"{a.name}.json"), "w"))
        print(f"-> {a.name}: {arr.shape}; {n_none} series without bone boxes (missing ROI boxes)")


if __name__ == "__main__":
    main()
