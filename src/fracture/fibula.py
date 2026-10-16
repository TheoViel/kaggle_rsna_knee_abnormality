"""A ninth ROI: the fibular head, derived from the U-Net's ROI boxes, and its crops; run from src/.

    python -m fracture.fibula --check                   # crops of studies with a reported fibular-head fracture
    python -m fracture.fibula --build                   # output/crops/f9_s1n8.{npy,csv,json}: f_s1n8 + the fibula crop
    python -m fracture.fibula --build --base fastmri_s1n8 --name fastmri9_s1n8 \
        --boxes ../output/fastmri/rois.csv --cache_dir ../output/fastmri/cache_384      # external data

The ROI U-Net has no fibula class, and ~6% of reported fractures (fibular head / neck / fibula) lie outside
every ROI box. The fibular head sits posterolateral, just below the lateral tibial plateau. Its box is placed
from the series' own boxes, so that it works whatever the stored orientation:
  lateral    the lateral meniscus side (relative to the medial meniscus) along the left-right axis
             (slices for sagittal series, columns for coronal);
  inferior   from the PF box (patella) towards the joint line (rows);
  posterior  from the PF box towards the Baker's box along the anterior-posterior axis (columns for sagittal,
             slices for coronal).
Box (relative to the lateral meniscus box, LM, and the lateral OA box height h):
  left-right   from the LM centre to past its outer edge (the lateral edge of the tibia);
  rows         from just below the joint line (LM bottom) down 0.8 h;
  front-back   the posterior half of the LM, extended past its posterior edge.
Crops: exactly the f_s1n8 sampling (8 strided slices, 128 px, margin 0.25), same rows as f_s1n8.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from crops.prepare import BOX, CROP_DIR, crop_box
from params import CACHE_DIR, OUTPUT_FOLDER

BOXES = os.path.join(OUTPUT_FOLDER, "roi", "labels", "round3_pred.csv")
# Through-plane sampling of the extra crops: the base crop set's (its .json), set in main(); f_s1n8 by default.
CROP_KW = dict(depth=8, z_mode="strided", stride=1)
NEEDED = ("Medial Meniscus", "Lateral Meniscus", "Lateral OA", "PF OA", "Baker's")


def _c(b, a):
    return (b[f"{a}0"] + b[f"{a}1"]) / 2


def fibula_box(bx, plane, rows, cols, n_slices):
    """ROI boxes of one sagittal / coronal series ({class: box}) -> the fibular-head box, or None."""
    if not set(NEEDED) <= set(bx):
        return None
    mm, lm, loa, pf, bk = (bx[c] for c in NEEDED)
    lr, ap = ("z", "x") if plane == "Sagittal" else ("x", "z")
    lim = {"z": n_slices - 1, "y": rows - 1, "x": cols - 1}
    lat = 1 if _c(lm, lr) > _c(mm, lr) else -1               # direction of lateral along the left-right axis
    inf = 1 if _c(lm, "y") > _c(pf, "y") else -1              # direction of inferior along rows
    post = 1 if _c(bk, ap) > _c(pf, ap) else -1               # direction of posterior along the AP axis
    out = {}
    # left-right: LM centre -> 0.5 x LM width past its outer edge
    w = lm[f"{lr}1"] - lm[f"{lr}0"] + 1
    outer = lm[f"{lr}1"] if lat > 0 else lm[f"{lr}0"]
    a, b = _c(lm, lr), outer + lat * 0.5 * w
    out[f"{lr}0"], out[f"{lr}1"] = min(a, b), max(a, b)
    # rows: from the joint line (LM bottom edge) + 0.05 h to + 0.8 h, h = lateral OA box height
    h = loa["y1"] - loa["y0"] + 1
    joint = lm["y1"] if inf > 0 else lm["y0"]
    a, b = joint + inf * 0.05 * h, joint + inf * 0.8 * h
    out["y0"], out["y1"] = min(a, b), max(a, b)
    # front-back: LM centre -> 0.3 x LM length past its posterior edge
    d = lm[f"{ap}1"] - lm[f"{ap}0"] + 1
    back = lm[f"{ap}1"] if post > 0 else lm[f"{ap}0"]
    a, b = _c(lm, ap), back + post * 0.3 * d
    out[f"{ap}0"], out[f"{ap}1"] = min(a, b), max(a, b)
    return {k: int(np.clip(round(out[k]), 0, lim[k[0]])) for k in BOX}


def boxes_per_series(path=BOXES):
    pred = pd.read_csv(path)
    pred = pred[pred.z0.notna()]
    out = {}
    for r in pred.to_dict("records"):
        out.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in BOX}
    return out


def _crop(job):
    import torch

    torch.set_num_threads(1)
    r, box, cache_dir, kw = job if len(job) == 4 else (*job, CROP_KW)
    if box is None:
        return np.zeros((kw["depth"], 128, 128), np.uint8)
    cache = np.load(os.path.join(cache_dir, f"{r['SeriesInstanceUID']}.npy"))
    return crop_box(cache, box, r["n_slices"], r["rows"], r["cols"], r["side"] == "L", r["Anatomical_Plane"] == "Sagittal",
                    size=128, margin=0.25, **kw)


def jobs(info, bx, cache_dir=CACHE_DIR):
    return [(r, fibula_box(bx.get(r["SeriesInstanceUID"], {}), r["Anatomical_Plane"], r["rows"], r["cols"], r["n_slices"]),
             cache_dir, dict(CROP_KW)) for r in info.to_dict("records")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--base", default="f_s1n8", help="8-ROI crop set to extend")
    p.add_argument("--name", default="f9_s1n8")
    p.add_argument("--boxes", default=BOXES)
    p.add_argument("--cache_dir", default=CACHE_DIR)
    a = p.parse_args()
    info = pd.read_csv(os.path.join(CROP_DIR, f"{a.base}.csv"))
    meta = json.load(open(os.path.join(CROP_DIR, f"{a.base}.json")))
    CROP_KW.update({k: meta[k] for k in ("depth", "z_mode", "stride") if k in meta})
    bx = boxes_per_series(a.boxes)
    if a.check:
        from PIL import Image, ImageDraw

        loc = json.load(open(os.path.join(OUTPUT_FOLDER, "locations", "fracture_locations.json")))["studies"]
        fib = [r["StudyInstanceUID"] for r in loc if isinstance(r["locations"], list)
               and any(l["structure"] == "fibular head" for l in r["locations"])]
        sel = info[info.StudyInstanceUID.isin(fib)].groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(1).head(16)
        strips = []
        for j, c in zip(jobs(sel, bx, a.cache_dir), map(_crop, jobs(sel, bx, a.cache_dir))):
            img = Image.fromarray(np.concatenate(list(c), 1)).convert("RGB")
            ImageDraw.Draw(img).text((3, 3), f"{j[0]['StudyInstanceUID'][-8:]} {j[0]['Anatomical_Plane']}", fill=(255, 255, 0))
            strips.append(np.array(img))
        Image.fromarray(np.concatenate(strips, 0)).save(a.out)
        print(f"{len(strips)} series -> {a.out}")
    if a.build:
        base = np.load(os.path.join(CROP_DIR, f"{a.base}.npy"), mmap_mode="r")
        arr = np.lib.format.open_memmap(os.path.join(CROP_DIR, f"{a.name}.npy"), mode="w+", dtype=np.uint8,
                                        shape=(base.shape[0], 9, *base.shape[2:]))
        js = jobs(info, bx, a.cache_dir)
        n_none = sum(b is None for _, b, _, _ in js)
        with ProcessPoolExecutor(96) as ex:
            for i, c in enumerate(ex.map(_crop, js, chunksize=16)):
                arr[i, :8] = base[i]
                arr[i, 8] = c
        arr.flush()
        info.to_csv(os.path.join(CROP_DIR, f"{a.name}.csv"), index=False)
        meta = json.load(open(os.path.join(CROP_DIR, f"{a.base}.json")))
        json.dump({**meta, "fibula": True}, open(os.path.join(CROP_DIR, f"{a.name}.json"), "w"))
        print(f"-> {a.name}: {arr.shape}; {n_none} series without a fibula box (missing ROI boxes)")


if __name__ == "__main__":
    main()
