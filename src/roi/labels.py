"""Box labels for the ROI U-Net: round 0 from attention, later rounds merged with
the U-Net's own predictions. Run from src/.

    python -m roi.labels attention --tag ens --out round0 --drop_classes "Baker's"
    python -m roi.labels merge --prev round0 --pred round0_pred --out round1
    python -m roi.labels bakers --tag ens --pred round1_pred --out bakers0

Round 0 (`attention`):
  - a candidate box per (series, class) from the attention maps of `roi.extract`
    (`roi.boxes.grid_box`, "zrun" mode);
  - localised = compact: in-plane box area <= `--max_area` of the slice and slice
    run <= `--max_z_frac` of the series. Absolute caps, not per-class quantiles:
    for classes whose maps are mostly diffuse (MCL, ACL, medial OA) a quantile
    would force broad boxes into the labels, and partial labels are fine;
  - cross-view check: a box is dropped when the other plane of the same study has
    localised boxes for the class and none of them intersects it in patient
    space. Where only one view has a box, nothing can be checked and it is kept.
Everything else is unlabelled, not negative: the loss ignores it.

Later rounds (`merge`): see `merge_labels`.

Output: output/roi/labels/<name>.csv, one row per labelled (series, class):
SeriesInstanceUID, StudyInstanceUID, plane, class, z0..x1 (original voxels,
inclusive), source, score.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import OUTPUT_FOLDER
from roi.boxes import cross_view, grid_box, localisation, to_voxels
from roi.extract import ATTN_DIR, ROI_CLASSES

LABEL_DIR = os.path.join(OUTPUT_FOLDER, "roi", "labels")
BOX_COLS = ["z0", "z1", "y0", "y1", "x0", "x1"]


def _candidates(path):
    uid = os.path.basename(path)[:-4]
    d = np.load(path)
    grid = d["xy"].shape[-1]
    rows, cols = int(d["rows"]), int(d["cols"])
    out = []
    for k, c in enumerate(ROI_CLASSES):
        b = grid_box(d["xy"][k], d["z"][k])
        out.append(dict(SeriesInstanceUID=uid, **{"class": c}, purity=b["purity"],
                        z_peak=float(d["z"][k].max() * len(d["z"][k])),
                        **localisation(b, len(d["z"][k]), grid, rows, cols),
                        **to_voxels(b, d["slices"], rows, cols, grid)))
    return out


def attention_candidates(tag):
    path = os.path.join(ATTN_DIR, tag, "candidates.csv")
    files = sorted(glob.glob(os.path.join(ATTN_DIR, tag, "maps", "*.npz")))
    if os.path.exists(path) and len(pd.read_csv(path, usecols=["SeriesInstanceUID"]).SeriesInstanceUID.unique()) == len(files):
        return pd.read_csv(path)
    with ProcessPoolExecutor(64) as ex:
        rows = [r for part in ex.map(_candidates, files, chunksize=32) for r in part]
    cand = pd.DataFrame(rows)
    info = pd.read_csv(os.path.join(OUTPUT_FOLDER, "series_index.csv"),
                       usecols=["SeriesInstanceUID", "StudyInstanceUID", "Anatomical_Plane"])
    cand = cand.merge(info.rename(columns={"Anatomical_Plane": "plane"}), on="SeriesInstanceUID")
    cand.to_csv(path, index=False)
    return cand


def drop_disagreeing(boxes):
    """Remove boxes that every box of the other plane (same study, class) misses."""
    from roi.boxes import iou, world_box

    keep = np.ones(len(boxes), bool)
    boxes = boxes.reset_index(drop=True)
    for (_, _), g in boxes.groupby(["StudyInstanceUID", "class"]):
        planes = set(g.plane)
        if len(planes) < 2:
            continue
        world = {i: world_box(r.SeriesInstanceUID, r.z0, r.z1, r.y0, r.y1, r.x0, r.x1)
                 for i, r in g.iterrows()}
        for i, r in g.iterrows():
            others = [j for j, s in g.iterrows() if s.plane != r.plane]
            if not any(iou(world[i], world[j]) > 0 for j in others):
                keep[i] = False
    return boxes[keep]


def round0(tag, max_area, max_z_frac, drop_classes=()):
    cand = attention_candidates(tag)
    cand = cand[~cand["class"].isin(drop_classes)]
    loc = cand[(cand.area <= max_area) & (cand.z_frac <= max_z_frac)]
    kept = drop_disagreeing(loc)
    print(f"localised {len(loc)} of {len(cand)}, kept after cross-view check {len(kept)} "
          f"({len(loc) - len(kept)} dropped as disagreeing)")
    return kept.assign(source="attention", score=1 - kept.area / max_area)


def summary(labels, n_series):
    by = labels.groupby("class").size().reindex(ROI_CLASSES).fillna(0).astype(int)
    print(f"{len(labels)} labels on {labels.SeriesInstanceUID.nunique()} / {n_series} series; per class:")
    print("  " + "  ".join(f"{c}: {n}" for c, n in by.items()))
    cv = cross_view(labels)
    if len(cv):
        print(f"  cross-view pairs {len(cv)}, overlap {cv.overlap.mean():.3f}, IoU {cv.iou.mean():.3f}")


def merge_labels(prev, pred, extra, min_conf, min_iou, series, drop_classes=(), max_volume=0.25):
    """Next round's labels from the previous ones, extra label sets and the U-Net's predictions.

    For each (series, class), in order:
      - a predicted box that another series of the study confirms -- patient-space
        IoU >= `min_iou` with that series' box, in the other plane or the same
        plane (`roi.consensus.add_confidence`) -- becomes the label, replacing
        the previous one: a box two acquisitions agree on beats one from a single
        view's attention or from a prior;
      - otherwise the previous label (or an `extra` one, e.g. the MCL prior)
        stays;
      - medial / lateral pairs ordered the wrong way for the knee are dropped
        (`roi.laterality.wrong_side`), and so are meniscus / compartment pairs
        whose meniscus centre is outside its compartment's box -- each meniscus
        lies in its tibiofemoral compartment (`roi.laterality.pair_consistency`);
      - what is still unlabelled in a study with labels for the class elsewhere
        is filled by registration (`roi.consensus.fill_by_registration`): every
        series has every ROI, so a gap is a failure to recover, not an absence.
    Only classes the predicting model was trained on -- those in `prev`, before
    `extra` is added -- are taken from `pred`: an untrained output predicts
    near-whole-volume boxes that trivially agree across views. Boxes covering
    more than `max_volume` of their series are never accepted, whatever the
    class. `drop_classes` removes a class from `prev` and `pred` altogether, so
    that an `extra` set replaces it.
    """
    from roi.consensus import _world, add_confidence, fill_by_registration
    from roi.laterality import wrong_side

    prev = prev[~prev["class"].isin(drop_classes)]
    trained = set(prev["class"])
    prev = pd.concat([prev] + extra, ignore_index=True).drop_duplicates(["SeriesInstanceUID", "class"])
    pred = pred[pred["class"].isin(trained)].copy()
    shapes = series.set_index("SeriesInstanceUID")[["n_slices", "rows", "cols"]].loc[pred.SeriesInstanceUID]
    vol = ((pred.z1 - pred.z0 + 1).to_numpy() / shapes.n_slices.to_numpy()
           * (pred.y1 - pred.y0 + 1).to_numpy() / shapes.rows.to_numpy()
           * (pred.x1 - pred.x0 + 1).to_numpy() / shapes.cols.to_numpy())
    pred = pred[vol <= max_volume]
    if "p_mean" not in pred:
        pred["p_mean"] = pred.conf
    pred = add_confidence(pred)
    for c in BOX_COLS:
        pred[c] = pred[c].astype(int)
    ok = (pred.conf >= min_conf) & ((pred.iou_other_plane >= min_iou) | (pred.iou_same_plane >= min_iou))
    new = pred[ok].assign(source="unet", score=lambda d: d.confidence)
    key = ["SeriesInstanceUID", "class"]
    old = prev.merge(new[key], on=key, how="left", indicator=True)
    old = old[old._merge == "left_only"].drop(columns="_merge")
    cols = ["SeriesInstanceUID", "StudyInstanceUID", "plane", "class", "source", "score"] + BOX_COLS
    out = pd.concat([new[cols], old[cols]], ignore_index=True)
    print(f"U-Net boxes confirmed by another series: {len(new)} "
          f"({int(new.merge(prev[key], on=key).shape[0])} replace a previous label); "
          f"previous / extra labels kept: {len(old)}")

    bad = wrong_side(out)
    out = out[~bad].reset_index(drop=True)
    print(f"dropped {int(bad.sum())} medial / lateral labels on the wrong side of the knee")
    from roi.laterality import pair_consistency

    _, inconsistent = pair_consistency(out)
    out = out.drop(index=inconsistent).reset_index(drop=True)
    print(f"dropped {len(inconsistent)} meniscus / compartment labels whose meniscus is outside its compartment")

    out["world"] = _world(out)
    out["confidence"] = out.score
    out, n_filled = fill_by_registration(out, series, trusted=pd.Series(True, index=out.index))
    out["score"] = out.score.fillna(out.confidence)
    print(f"filled {n_filled} (series, class) gaps by registration from the study's other series")
    return out[cols]


def bakers_labels(tag, pred, max_area, max_z_frac, margin):
    """Baker's boxes from attention, kept only if behind the knee.

    Same compactness caps and cross-view check as round 0, plus anatomy: the
    box centre must lie at least `margin` mm posterior (+y in patient LPS) to the
    centres of the series' predicted femorotibial compartment boxes.
    """
    from roi.boxes import world_box

    cand = attention_candidates(tag)
    cand = cand[cand["class"] == "Baker's"]
    loc = drop_disagreeing(cand[(cand.area <= max_area) & (cand.z_frac <= max_z_frac)])
    comp = pred[pred["class"].isin(["Medial OA", "Lateral OA"]) & pred.z0.notna()]
    comp_y = {}
    for r in comp.itertuples():
        lo, hi = world_box(r.SeriesInstanceUID, int(r.z0), int(r.z1), int(r.y0), int(r.y1), int(r.x0), int(r.x1))
        comp_y.setdefault(r.SeriesInstanceUID, []).append((lo[1] + hi[1]) / 2)
    keep = []
    for i, r in loc.iterrows():
        if r.SeriesInstanceUID not in comp_y:
            continue
        lo, hi = world_box(r.SeriesInstanceUID, r.z0, r.z1, r.y0, r.y1, r.x0, r.x1)
        if (lo[1] + hi[1]) / 2 >= max(comp_y[r.SeriesInstanceUID]) + margin:
            keep.append(i)
    out = loc.loc[keep].assign(source="attention", score=lambda d: 1 - d.area / max_area)
    print(f"Baker's: {len(cand)} maps, {len(loc)} compact and agreeing across views, "
          f"{len(out)} behind the knee")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("attention")
    a0.add_argument("--tag", default="ens")
    a0.add_argument("--out", default="round0")
    a0.add_argument("--max_area", type=float, default=0.08)
    a0.add_argument("--max_z_frac", type=float, default=0.5)
    a0.add_argument("--drop_classes", default="", help="comma-separated classes left unlabelled")
    m = sub.add_parser("merge")
    m.add_argument("--prev", required=True)
    m.add_argument("--pred", required=True)
    m.add_argument("--extra", default="", help="comma-separated extra label sets, e.g. mcl_prior,bakers0")
    m.add_argument("--out", required=True)
    m.add_argument("--min_conf", type=float, default=0.5)
    m.add_argument("--min_iou", type=float, default=0.3)
    m.add_argument("--drop_classes", default="", help="comma-separated classes to remove from prev and pred")
    bk = sub.add_parser("bakers")
    bk.add_argument("--tag", default="ens")
    bk.add_argument("--pred", default="round1_pred")
    bk.add_argument("--out", default="bakers0")
    bk.add_argument("--max_area", type=float, default=0.08)
    bk.add_argument("--max_z_frac", type=float, default=0.5)
    bk.add_argument("--margin", type=float, default=10.0)
    a = p.parse_args()

    os.makedirs(LABEL_DIR, exist_ok=True)
    n_series = len(pd.read_csv(os.path.join(OUTPUT_FOLDER, "roi", "volumes.csv")))
    if a.cmd == "attention":
        labels = round0(a.tag, a.max_area, a.max_z_frac, [c for c in a.drop_classes.split(",") if c])
    elif a.cmd == "bakers":
        pred = pd.read_csv(os.path.join(LABEL_DIR, f"{a.pred}.csv"))
        labels = bakers_labels(a.tag, pred, a.max_area, a.max_z_frac, a.margin)
    else:
        read = lambda n: pd.read_csv(os.path.join(LABEL_DIR, f"{n}.csv"))  # noqa: E731
        series = pd.read_csv(os.path.join(OUTPUT_FOLDER, "roi", "volumes.csv"))
        labels = merge_labels(read(a.prev), read(a.pred), [read(n) for n in a.extra.split(",") if n],
                              a.min_conf, a.min_iou, series,
                              drop_classes=[c for c in a.drop_classes.split(",") if c])
    cols = ["SeriesInstanceUID", "StudyInstanceUID", "plane", "class", "source", "score"] + BOX_COLS
    labels[cols].to_csv(os.path.join(LABEL_DIR, f"{a.out}.csv"), index=False)
    summary(labels, n_series)
    print(f"-> {os.path.join(LABEL_DIR, a.out + '.csv')}")


if __name__ == "__main__":
    main()
