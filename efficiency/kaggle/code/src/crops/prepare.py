"""ROI crops for the finding classifiers, run from src/.

    python -m crops.prepare --boxes round3_pred --name r3
    python -m crops.prepare --boxes round3_axial --name r3_axial --volumes volumes_axial

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

Boxes come from the ROI U-Net's predictions (`--boxes`, output/roi/labels/<name>.csv),
which is also what inference produces, so training and test crops match.

Output: output/crops/<name>.npy (N, K, DEPTH, SIZE, SIZE) uint8, memory-mappable,
and <name>.csv (row -> series, plane, study).
"""

import argparse
import json
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


def crop_window(n_c, cs, box, n_slices, rows, cols, depth=DEPTH, margin=MARGIN, min_side=MIN_SIDE,
                z_mode="uniform", stride=1):
    """Where `crop_box` reads in a (n_c, cs, cs) cache: slice positions, window origin, window side.

    z_mode, through-plane sampling (positions in original slices, mapped to cache slices):
      uniform  `depth` positions spread evenly over the box's slices widened by one, nearest slice;
      strided  `depth` slices `stride` apart centred on the box's centre (regardless of its extent);
      resize   the uniform positions, linearly interpolated between slices (float positions).
    """
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
    if "x_lo" in box or "x_hi" in box:
        # Compartment crops (oa_midline): slide the window so it stays on its side of the
        # intercondylar midline; it keeps its size and gains periphery instead.
        lo = (box["x_lo"] + ox) * f if "x_lo" in box else -np.inf
        hi = (box["x_hi"] + 1 + ox) * f if "x_hi" in box else np.inf
        if x0 + n > hi:
            x0 = int(round(hi - n))
        if x0 < lo:
            x0 = int(round(lo))
    # Slices: box range widened by one, mapped to cache indices.
    zc = (lambda z: z * (n_c - 1) / max(n_slices - 1, 1)) if n_c != n_slices else (lambda z: z)
    if z_mode == "strided":
        start = np.floor((box["z0"] + box["z1"]) / 2 - (depth - 1) * stride / 2 + 0.5)
        zs = np.clip(np.round(zc(start + np.arange(depth) * stride)).astype(int), 0, n_c - 1)
        return zs, y0, x0, n
    za, zb = zc(max(box["z0"] - 1, 0)), zc(min(box["z1"] + 1, n_slices - 1))
    if z_mode == "resize":
        return np.clip(np.linspace(za, zb, depth), 0, n_c - 1), y0, x0, n
    zs = np.clip(np.round(np.linspace(za, zb, depth)).astype(int), 0, n_c - 1)
    return zs, y0, x0, n


def crop_box(cache, box, n_slices, rows, cols, flip, sagittal, depth=DEPTH, size=SIZE,
             margin=MARGIN, min_side=MIN_SIDE, z_mode="uniform", stride=1):
    """One (depth, size, size) uint8 crop of a (n_c, 384, 384) cache, see module doc."""
    import torch
    import torch.nn.functional as F

    n_c, cs = cache.shape[0], cache.shape[1]
    zs, y0, x0, n = crop_window(n_c, cs, box, n_slices, rows, cols, depth, margin, min_side, z_mode, stride)
    # Zero-padded window so crops reaching past the edge keep their geometry.
    win = np.zeros((depth, n, n), np.float32)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + n, cs), min(x0 + n, cs)
    if ye > ys and xe > xs:
        if zs.dtype.kind == "f":          # resize: linear between the two nearest slices
            lo = np.floor(zs).astype(int)
            hi, w = np.minimum(lo + 1, n_c - 1), (zs - lo)[:, None, None]
            sub = (1 - w) * cache[lo][:, ys:ye, xs:xe] + w * cache[hi][:, ys:ye, xs:xe]
        else:
            sub = cache[zs][:, ys:ye, xs:xe]
        win[:, ys - y0:ye - y0, xs - x0:xe - x0] = sub
    t = torch.from_numpy(win)[:, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", antialias=True, align_corners=False)[:, 0]
    out = t.round().clamp(0, 255).to(torch.uint8).numpy()
    if flip:
        out = out[::-1] if sagittal else out[:, :, ::-1]
    return np.ascontiguousarray(out)


def capsule_boxes(pred, expand=0.0, expand_up=None, shape=None, classes=None):
    """Per series, one "Capsule" box: the union of its ROI boxes, widened in-plane by `expand`
    x its height / width on each side (`expand_up` for the top, superior edge), clipped to
    the image (`shape`: series -> (rows, cols)). Slices: the union's range, unchanged.

    Effusion and synovitis have no box of their own, but they live in the joint capsule,
    which the eight ROIs (menisci, compartments, ligaments, patellofemoral joint, Baker's)
    outline; the suprapatellar recess reaches above the patella, hence the separate
    superior margin.
    """
    up = expand if expand_up is None else expand_up
    if classes:                       # union of these ROIs only (e.g. the tibiofemoral joint)
        pred = pred[pred["class"].isin(classes)]
    out = {}
    for uid, g in pred.groupby("SeriesInstanceUID"):
        b = dict(z0=int(g.z0.min()), z1=int(g.z1.max()), y0=g.y0.min(), y1=g.y1.max(), x0=g.x0.min(), x1=g.x1.max())
        h, w = b["y1"] - b["y0"] + 1, b["x1"] - b["x0"] + 1
        rows, cols = shape[uid]
        b.update(y0=int(max(0, round(b["y0"] - up * h))), y1=int(min(rows - 1, round(b["y1"] + expand * h))),
                 x0=int(max(0, round(b["x0"] - expand * w))), x1=int(min(cols - 1, round(b["x1"] + expand * w))))
        out[uid] = {"Capsule": b}
    return out


def oa_midline(bx, clip=False):
    """Medial / lateral OA boxes of a coronal or axial series -> the same boxes with the column
    limit of their side of the knee: the midline halfway between the two menisci boxes' inner
    edges. Columns run left-right in both planes (sagittal series need nothing: left-right is
    the slice axis, and their OA crops stay inside their compartment)."""
    if not {"Medial Meniscus", "Lateral Meniscus"} <= set(bx):
        return bx
    mm, lm = bx["Medial Meniscus"], bx["Lateral Meniscus"]
    med_left = (mm["x0"] + mm["x1"]) < (lm["x0"] + lm["x1"])
    mid = (mm["x1"] + lm["x0"]) / 2 if med_left else (lm["x1"] + mm["x0"]) / 2
    out = dict(bx)
    for c, left in (("Medial OA", med_left), ("Lateral OA", not med_left)):
        if c in out:
            b = {**out[c], **({"x_hi": mid} if left else {"x_lo": mid})}
            if clip:                  # the box itself stops at the midline too (smaller, tighter crop)
                if left:
                    b["x1"] = int(max(b["x0"], min(b["x1"], np.floor(mid))))
                else:
                    b["x0"] = int(min(b["x1"], max(b["x0"], np.ceil(mid))))
            out[c] = b
    return out


def notch_box(bx, plane, shape, n_slices, union=False):
    """The intercondylar notch, where the ACL runs, from the two meniscus boxes (reliable),
    instead of the U-Net's ACL box (learned from attention maps that favoured the bone bruise
    on the posterolateral tibial plateau). `union`: joined with the U-Net ACL box (keeps the
    bruise as well).

    Sagittal (left-right = slices): the slices between the two menisci, at least 5; in-plane
    the menisci's anterior-posterior extent, from half their height above the joint line (the
    notch roof) to a quarter below (tibial spines). Coronal (left-right = columns): the columns
    between the menisci's inner edges, at least 20% of the joint's width; rows as above;
    slices: the menisci's anterior-posterior range."""
    if not {"Medial Meniscus", "Lateral Meniscus"} <= set(bx):
        return bx
    mm, lm = bx["Medial Meniscus"], bx["Lateral Meniscus"]
    lr = "z" if plane == "Sagittal" else "x"
    a, b = (mm, lm) if (mm[f"{lr}0"] + mm[f"{lr}1"]) < (lm[f"{lr}0"] + lm[f"{lr}1"]) else (lm, mm)
    lo, hi = a[f"{lr}1"], b[f"{lr}0"]                     # inner edges
    joint = b[f"{lr}1"] - a[f"{lr}0"] + 1
    half = max((hi - lo) / 2, 2.5 if lr == "z" else 0.10 * joint)
    mid = (lo + hi) / 2
    y0, y1 = min(mm["y0"], lm["y0"]), max(mm["y1"], lm["y1"])
    h = y1 - y0 + 1
    out = {"y0": int(max(0, round(y0 - 0.5 * h))), "y1": int(min(shape[0] - 1, round(y1 + 0.25 * h)))}
    lim = n_slices - 1 if lr == "z" else shape[1] - 1
    out[f"{lr}0"], out[f"{lr}1"] = int(max(0, np.floor(mid - half))), int(min(lim, np.ceil(mid + half)))
    other = "x" if lr == "z" else "z"
    out[f"{other}0"], out[f"{other}1"] = min(mm[f"{other}0"], lm[f"{other}0"]), max(mm[f"{other}1"], lm[f"{other}1"])
    if union and "ACL" in bx:
        out = {k: (min if k.endswith("0") else max)(out[k], bx["ACL"][k]) for k in BOX}
    return {**bx, "ACL": out}


def compartment_box(oa, men, extend, shape):
    """A tibiofemoral compartment: the OA box joined with the meniscus box of the same side,
    extended by `extend` x its height above and below to take in the femoral condyle and the
    tibial plateau (osteophytes, subchondral bone), clipped to the image."""
    b = {k: (min if k.endswith("0") else max)(oa[k], men[k]) for k in BOX}
    h = b["y1"] - b["y0"] + 1
    b["y0"] = int(max(0, round(b["y0"] - extend * h)))
    b["y1"] = int(min(shape[0] - 1, round(b["y1"] + extend * h)))
    return b


def _series(args):
    import torch

    torch.set_num_threads(1)
    r, boxes, flip, z, classes, size, margin, min_side, cache_dir = args
    cache = np.load(os.path.join(cache_dir, f"{r['SeriesInstanceUID']}.npy"))
    out = np.zeros((len(classes), z["depth"], size, size), np.uint8)
    for k, c in enumerate(classes):
        if c in boxes:
            out[k] = crop_box(cache, boxes[c], r["n_slices"], r["rows"], r["cols"], flip,
                              r["Anatomical_Plane"] == "Sagittal", size=size, margin=margin, min_side=min_side, **z)
    return out


def main():
    from roi.labels import LABEL_DIR
    from roi.laterality import series_side

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxes", default="round3_pred")
    p.add_argument("--name", default="r3")
    p.add_argument("--volumes", default="volumes", help="output/roi/<volumes>.csv (volumes or volumes_axial), or a path to a .csv")
    p.add_argument("--depth", type=int, default=DEPTH, help="frames per crop")
    p.add_argument("--z_mode", default="uniform", choices=["uniform", "strided", "resize"])
    p.add_argument("--stride", type=int, default=1, help="slices between frames (strided)")
    p.add_argument("--size", type=int, default=SIZE, help="crop side, px")
    p.add_argument("--capsule", type=float, default=None,
                   help="one Capsule crop per series instead of the 8 ROIs, the ROI union widened by this fraction")
    p.add_argument("--capsule_up", type=float, default=None, help="superior widening (default: --capsule)")
    p.add_argument("--union", default="", help="with --capsule: comma-separated ROIs whose union is the box "
                                               "(default: all 8)")
    p.add_argument("--margin", type=float, default=MARGIN, help="in-plane context around ROI boxes (x side)")
    p.add_argument("--min_side", type=float, default=MIN_SIDE, help="smallest crop window, cache px")
    p.add_argument("--cache_dir", default=CACHE_DIR, help="384 px caches (default: the competition's)")
    p.add_argument("--side_csv", default="", help="SeriesInstanceUID, side (R / L) instead of the side classifier's")
    p.add_argument("--oa_midline", default="", choices=["", "slide", "clip"],
                   help="coronal / axial medial and lateral OA crops stop at the intercondylar midline "
                        "(slide: the window moves outward; clip: the box is also cut at the midline)")
    p.add_argument("--acl_notch", default="", choices=["", "notch", "union"],
                   help="replace the ACL box by the intercondylar notch from the menisci boxes "
                        "(union: notch joined with the U-Net ACL box)")
    p.add_argument("--compartment", type=float, default=None,
                   help="medial / lateral OA boxes: union with the same side's meniscus box, then extended "
                        "by this fraction of its height above and below (femur, tibia)")
    a = p.parse_args()
    z = dict(depth=a.depth, z_mode=a.z_mode, stride=a.stride)
    os.makedirs(CROP_DIR, exist_ok=True)
    vols = pd.read_csv(a.volumes if a.volumes.endswith(".csv") else os.path.join(OUTPUT_FOLDER, "roi", f"{a.volumes}.csv"))
    pred = pd.read_csv(a.boxes if a.boxes.endswith(".csv") else os.path.join(LABEL_DIR, f"{a.boxes}.csv"))
    pred = pred[pred.z0.notna()]
    boxes = {}
    if a.capsule is not None:
        # The box is the whole field of view of interest: square crop of its larger side, no margin.
        classes, margin = ["Capsule"], 0.0
        shape = {r.SeriesInstanceUID: (r.rows, r.cols) for r in vols.itertuples()}
        union = [c for c in a.union.split(",") if c]
        boxes = capsule_boxes(pred, a.capsule, a.capsule_up, shape, union or None)
        z.update(capsule=a.capsule, capsule_up=a.capsule_up, **({"union": a.union} if union else {}))
    else:
        classes, margin = list(ROI_CLASSES), a.margin
        for r in pred.to_dict("records"):
            boxes.setdefault(r["SeriesInstanceUID"], {})[r["class"]] = {k: int(r[k]) for k in BOX}
        if a.compartment is not None:
            shape = {r.SeriesInstanceUID: (r.rows, r.cols) for r in vols.itertuples()}
            for uid, bx in boxes.items():
                for oa, men in (("Medial OA", "Medial Meniscus"), ("Lateral OA", "Lateral Meniscus")):
                    if oa in bx and men in bx and uid in shape:
                        bx[oa] = compartment_box(bx[oa], bx[men], a.compartment, shape[uid])
            z.update(compartment=a.compartment)
        if a.acl_notch:
            meta = {r.SeriesInstanceUID: r for r in vols.itertuples()}
            for uid in list(boxes):
                r = meta.get(uid)
                if r is not None and r.Anatomical_Plane in ("Sagittal", "Coronal"):
                    boxes[uid] = notch_box(boxes[uid], r.Anatomical_Plane, (r.rows, r.cols), r.n_slices,
                                           union=a.acl_notch == "union")
            z.update(acl_notch=a.acl_notch)
        if a.oa_midline:
            plane = dict(zip(vols.SeriesInstanceUID, vols.Anatomical_Plane))
            for uid in list(boxes):
                if plane.get(uid) in ("Coronal", "Axial"):
                    boxes[uid] = oa_midline(boxes[uid], clip=a.oa_midline == "clip")
            z.update(oa_midline=a.oa_midline)
        z.update(margin=a.margin)
    z["size"] = a.size
    side = series_side() if not a.side_csv else dict(pd.read_csv(a.side_csv)[["SeriesInstanceUID", "side"]].values)
    zc = {k: z[k] for k in ("depth", "z_mode", "stride")}
    jobs = [(r, boxes.get(r["SeriesInstanceUID"], {}), side.get(r["SeriesInstanceUID"]) == "L", zc, classes, a.size,
             margin, a.min_side, a.cache_dir) for r in vols.to_dict("records")]
    z["min_side"] = a.min_side
    arr = np.lib.format.open_memmap(os.path.join(CROP_DIR, f"{a.name}.npy"), mode="w+", dtype=np.uint8,
                                    shape=(len(jobs), len(classes), a.depth, a.size, a.size))
    with ProcessPoolExecutor(96) as ex:
        for i, c in enumerate(ex.map(_series, jobs, chunksize=8)):
            arr[i] = c
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    arr.flush()
    vols.assign(row=np.arange(len(vols)), side=vols.SeriesInstanceUID.map(side)).to_csv(
        os.path.join(CROP_DIR, f"{a.name}.csv"), index=False)
    with open(os.path.join(CROP_DIR, f"{a.name}.json"), "w") as f:      # sampling, for inference
        json.dump(z, f)
    print(f"-> {len(jobs)} series x {len(classes)} ROIs x {a.depth} x {a.size} x {a.size} ({z}) in {CROP_DIR}")


if __name__ == "__main__":
    main()
