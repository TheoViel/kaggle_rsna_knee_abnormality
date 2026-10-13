"""Bounding boxes of findings from the classifier's attention maps, run from src/.

    python attention_boxes.py --run_dir ../logs/cls_run/2026-10-08/4/            # infer + boxes
    python attention_boxes.py --run_dir ... --skip_inference --max_area 0.03     # re-threshold only

For every study (the radiologist-labelled ones by default, which the model never
trained on), every sagittal and coronal series and every class:

  1. Attention. The trained 2.5D model is run with a frame on every slice
     (`main_attention.StudyAttention`, dense mode), giving per frame an in-plane
     softmax map (`xy_attn`) and a weight over frames (`z_attn`).
  2. z extent. The contiguous run of frames around the z-attention peak whose
     weight is at least `--z_rel` times the peak.
  3. In-plane extent. The xy maps of those frames, averaged with their z weights,
     upsampled to the original slice grid; the smallest pixel set holding
     `--mass` of the attention; its connected components. The box is the
     bounding box of the component holding the most attention.
  4. Localised or not. A box is kept only if the attention is concentrated:
       - `purity`: share of the `--mass` region's attention in that main
         component (one blob, not several) >= `--min_purity`;
       - `area`: main component area / slice area <= `--max_area`;
       - `z_frac`: frames in the z run / frames in the series <= `--max_z_frac`.
     Diffuse attention means the model did not find the structure (or reads the
     finding from scattered evidence): no box, rather than a wrong one.

Axial series get no attention-derived boxes (the attention is only trusted on
sagittal and coronal). Their boxes are the localised sagittal and coronal boxes
of the same study mapped through world coordinates (`knee_registration`): the 8
corners of each box are sent to the axial voxel grid and bounded. Per plane the
box of the series the study attention weights most for that class is used, and
the sagittal and coronal boxes are intersected (sagittal is precise
antero-posteriorly and cranio-caudally, coronal left-right and cranio-caudally).
By default both views are required (`--axial_min_views 2`): a single view leaves
one axis known only to its slice run, and two independent views that intersect
are a consistency check -- a model error in one view (e.g. the wrong side for a
meniscus) gives disjoint boxes and no axial box.

Outputs, under `--out_dir` (default output/attention_boxes/<run date>_<run id>/):
  metrics.csv   every (series, class) map: localisation metrics, candidate box,
                study probability, series attention weight
  boxes.csv     the kept boxes, sagittal / coronal (source=attention) and axial
                (source=registered), in original voxel indices of each series:
                slices z0..z1, rows y0..y1, cols x0..x1, inclusive
  maps/<SeriesInstanceUID>.npz   the attention (for display): slices, xy_attn
                (K, F, h, w) float16, z_attn (K, F), rows, cols
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from params import CLASSES, OUTPUT_FOLDER

PLANES = ("Sagittal", "Coronal")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="../logs/cls_run/2026-10-08/4/")
    p.add_argument("--studies", default="gt", help="gt | comma-separated StudyInstanceUIDs")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--skip_inference", action="store_true",
                   help="reuse maps/ and only recompute metrics, boxes and registration")
    p.add_argument("--mass", type=float, default=0.5)
    p.add_argument("--z_rel", type=float, default=0.5)
    p.add_argument("--min_purity", type=float, default=0.8)
    p.add_argument("--max_area", type=float, default=0.03)
    p.add_argument("--max_z_frac", type=float, default=0.5)
    p.add_argument("--axial_min_views", type=int, default=2,
                   help="2: an axial box needs agreeing sagittal and coronal boxes; 1: either")
    p.add_argument("--refine", action="store_true",
                   help="rigid registration on top of the DICOM geometry for the axial transfer")
    return p.parse_args()


def default_out_dir(run_dir):
    parts = os.path.normpath(run_dir).split(os.sep)
    return os.path.join(OUTPUT_FOLDER, "attention_boxes", f"{parts[-2]}_{parts[-1]}")


# ------------------------------------------------------------------ inference


def run_inference(args, out_dir, study_ids):
    """Dense attention for every study; saves maps/ and returns study-level info."""
    from main_attention import StudyAttention

    sa = StudyAttention(args.run_dir)
    idx = sa.series.set_index("SeriesInstanceUID")
    os.makedirs(os.path.join(out_dir, "maps"), exist_ok=True)
    rows = []
    for i, uid in enumerate(study_ids):
        probs, recs = sa(uid, dense=True)
        for r in recs:
            plane = idx.loc[r["series_uid"], "Anatomical_Plane"]
            if plane not in PLANES:
                continue
            np.savez_compressed(
                os.path.join(out_dir, "maps", f"{r['series_uid']}.npz"),
                slices=r["slices"], xy_attn=r["xy_attn"], z_attn=r["z_attn"],
                rows=r["rows"], cols=r["cols"])
            for k, c in enumerate(CLASSES):
                rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": r["series_uid"],
                             "plane": plane, "class": c, "prob": float(probs[k]),
                             "series_attn": float(r["series_attn"][k])})
        print(f"{i + 1}/{len(study_ids)} {uid[-12:]}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- boxes


def z_run(z, rel):
    """Indices of the contiguous frames around the peak with weight >= rel * peak."""
    peak = int(np.argmax(z))
    keep = z >= rel * z[peak]
    lo = peak
    while lo > 0 and keep[lo - 1]:
        lo -= 1
    hi = peak
    while hi < len(z) - 1 and keep[hi + 1]:
        hi += 1
    return lo, hi


def smallest_mass_set(a, mass):
    flat = a.ravel() / a.sum()
    order = np.argsort(-flat)
    keep = np.zeros(flat.size, bool)
    keep[order[:np.searchsorted(np.cumsum(flat[order]), mass) + 1]] = True
    return keep.reshape(a.shape)


def localise(xy, z, slices, rows, cols, mass, z_rel):
    """Metrics and candidate box of one (series, class) attention map.

    xy (F, h, w) in-plane maps, z (F,) frame weights, slices (F,) original slice
    index of each frame.
    """
    from main_attention import to_original

    lo, hi = z_run(z, z_rel)
    w = z[lo:hi + 1] / z[lo:hi + 1].sum()
    avg = np.tensordot(w, xy[lo:hi + 1].astype(np.float32), axes=1)        # h w
    up = np.clip(to_original(avg, rows, cols), 0, None)
    region = smallest_mass_set(up, mass)
    lab, n = ndi.label(region, structure=np.ones((3, 3)))
    comp_mass = ndi.sum(up, lab, index=np.arange(1, n + 1))
    main = int(np.argmax(comp_mass)) + 1
    m = lab == main
    ys, xs = np.nonzero(m)
    return {
        "z0": int(slices[lo]), "z1": int(slices[hi]),
        "y0": int(ys.min()), "y1": int(ys.max()), "x0": int(xs.min()), "x1": int(xs.max()),
        "purity": float(comp_mass[main - 1] / comp_mass.sum()),
        "area": float(m.mean()),
        "z_frac": float((hi - lo + 1) / len(z)),
        "z_peak": float(z.max() * len(z)),          # 1 = uniform over frames
        "n_components": int(n),
    }


def compute_metrics(info, out_dir, args):
    rows = []
    for series_uid, g in info.groupby("SeriesInstanceUID", sort=False):
        d = np.load(os.path.join(out_dir, "maps", f"{series_uid}.npz"))
        for r in g.to_dict("records"):
            k = CLASSES.index(r["class"])
            met = localise(d["xy_attn"][k], d["z_attn"][k], d["slices"],
                           int(d["rows"]), int(d["cols"]), args.mass, args.z_rel)
            rows.append({**r, **met})
    m = pd.DataFrame(rows)
    m["localized"] = ((m.purity >= args.min_purity) & (m.area <= args.max_area)
                      & (m.z_frac <= args.max_z_frac))
    return m


# --------------------------------------------------------------- registration


def to_axial(box, src_img, ax_img, tx=None):
    """Axis-aligned box in ax_img voxel indices bounding the 8 corners of `box`."""
    corners = [(x, y, z) for x in (box.x0 - 0.5, box.x1 + 0.5)
               for y in (box.y0 - 0.5, box.y1 + 0.5) for z in (box.z0 - 0.5, box.z1 + 0.5)]
    pts = []
    for c in corners:
        p = src_img.TransformContinuousIndexToPhysicalPoint(c)
        if tx is not None:                      # tx maps axial (fixed) -> source (moving)
            p = tx.GetInverse().TransformPoint(p)
        pts.append(ax_img.TransformPhysicalPointToContinuousIndex(p))
    pts = np.array(pts)
    lo, hi = np.ceil(pts.min(0) - 0.5).astype(int), np.floor(pts.max(0) + 0.5).astype(int)
    size = np.array(ax_img.GetSize())
    lo, hi = np.maximum(lo, 0), np.minimum(hi, size - 1)
    if (hi < lo).any():
        return None
    return {"x0": lo[0], "x1": hi[0], "y0": lo[1], "y1": hi[1], "z0": lo[2], "z1": hi[2]}


def register_axial(boxes, series, refine=False, min_views=2):
    """Axial boxes from the localised sagittal / coronal ones, see module doc."""
    import knee_registration as kr

    geo = {}

    def geometry(uid):
        if uid not in geo:
            import SimpleITK as sitk

            r = series.loc[uid]
            g = kr.series_geometry(uid)
            img = sitk.Image([int(r.cols), int(r.rows), int(r.n_slices)], sitk.sitkUInt8)
            img.SetSpacing(g["spacing"]); img.SetDirection(g["direction"]); img.SetOrigin(g["origin"])
            geo[uid] = img
        return geo[uid]

    out = []
    for study, sb in boxes.groupby("StudyInstanceUID"):
        axials = series[(series.StudyInstanceUID == study) & (series.Anatomical_Plane == "Axial")]
        for ax_uid in axials.index:
            ax_img = kr.series_image(ax_uid) if refine else geometry(ax_uid)
            for c, cb in sb.groupby("class"):
                per_plane = {}
                for plane, pb in cb.groupby("plane"):
                    b = pb.sort_values("series_attn", ascending=False).iloc[0]
                    tx = None
                    if refine:
                        tx, _, _ = kr.rigid_refine(ax_img, kr.series_image(b.SeriesInstanceUID))
                    a = to_axial(b, geometry(b.SeriesInstanceUID), geometry(ax_uid), tx)
                    if a is not None:
                        per_plane[plane] = (a, b)
                if len(per_plane) < min_views:
                    continue
                boxes_ = [a for a, _ in per_plane.values()]
                fused = {k: max(a[k] for a in boxes_) if k.endswith("0") else min(a[k] for a in boxes_)
                         for k in boxes_[0]}
                if any(fused[f"{ax}1"] < fused[f"{ax}0"] for ax in "xyz"):
                    continue                     # the views disagree: no box
                src = per_plane[max(per_plane, key=lambda p: per_plane[p][1].series_attn)][1]
                out.append({"StudyInstanceUID": study, "SeriesInstanceUID": ax_uid,
                            "plane": "Axial", "class": c, "prob": src.prob,
                            "source": "registered",
                            "from_series": ";".join(b.SeriesInstanceUID for _, b in per_plane.values()),
                            **{k: int(v) for k, v in fused.items()}})
    return pd.DataFrame(out)


def main():
    args = parse_args()
    out_dir = args.out_dir or default_out_dir(args.run_dir)
    os.makedirs(out_dir, exist_ok=True)

    from knee_cls.data import load_tables

    studies, series = load_tables()
    ids = (studies.loc[studies.is_gt, "StudyInstanceUID"].tolist() if args.studies == "gt"
           else args.studies.split(","))

    info_path = os.path.join(out_dir, "series_info.csv")
    if args.skip_inference:
        info = pd.read_csv(info_path)
    else:
        info = run_inference(args, out_dir, ids)
        info.to_csv(info_path, index=False)

    metrics = compute_metrics(info, out_dir, args)
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    kept = metrics[metrics.localized].assign(source="attention", from_series="")
    axial = register_axial(kept, series.set_index("SeriesInstanceUID"), refine=args.refine,
                           min_views=args.axial_min_views)
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "class", "prob", "source",
            "from_series", "z0", "z1", "y0", "y1", "x0", "x1"]
    boxes = pd.concat([kept[cols + ["purity", "area", "z_frac"]], axial], ignore_index=True)
    boxes.to_csv(os.path.join(out_dir, "boxes.csv"), index=False)

    loc = metrics.groupby("class").localized.mean().reindex(CLASSES)
    print(f"-> {out_dir}")
    print(f"{len(metrics)} maps, {int(metrics.localized.sum())} localised; "
          f"{len(axial)} axial boxes from registration")
    print("share of sagittal/coronal maps localised, per class:")
    print("  " + "  ".join(f"{c[:10]} {v:.2f}" for c, v in loc.items()))


if __name__ == "__main__":
    main()
