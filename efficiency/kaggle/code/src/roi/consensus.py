"""Study-level consensus of ROI boxes: per-box confidence, and registration to fill gaps.

All sagittal and coronal series of a study share the patient coordinate system,
so a box can be compared with, and moved into, any other series of the study --
across planes (sagittal <-> coronal) and within one (two coronal series, e.g. a
PD fat-sat and a T1).

Per-box confidence (`add_confidence`), for U-Net predictions:
  - p_mean: mean U-Net probability over the box's component (the peak is ~1
    almost everywhere, the mean is high only for compact, sure predictions);
  - support: share of the study's other series whose box for the class
    intersects this one in patient space (NaN when the series is alone);
  - iou_other_plane / iou_same_plane: best patient-space IoU with a box of the
    other plane / of another series of the same plane;
  - confidence = p_mean * (0.5 + 0.5 * support), with support taken as 0 for a
    lone series: a box no other acquisition confirms is worth half as much.

Registration fill (`fill_by_registration`): every (series, class) of a study gets
a box; where it has none, or a weak one, the trusted boxes of the study's other
series are mapped into it. One trusted box per plane (the most confident); with
both planes, the two are intersected in patient space -- the sagittal box is
precise antero-posteriorly and cranio-caudally, the coronal one left-right and
cranio-caudally -- else the single box is used as is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from roi.boxes import iou, world_box
from roi.mcl_prior import world_to_voxel_box

BOX = ["z0", "z1", "y0", "y1", "x0", "x1"]


def _world(df):
    return [world_box(r.SeriesInstanceUID, int(r.z0), int(r.z1), int(r.y0), int(r.y1), int(r.x0), int(r.x1))
            for r in df.itertuples()]


def add_confidence(pred):
    pred = pred[pred.z0.notna()].copy().reset_index(drop=True)
    pred["world"] = _world(pred)
    support, other, same = {}, {}, {}
    for _, g in pred.groupby(["StudyInstanceUID", "class"]):
        for i, r in g.iterrows():
            rest = g.drop(index=i)
            ious = [(o.plane, iou(r.world, o.world)) for o in rest.itertuples()]
            support[i] = np.mean([v > 0 for _, v in ious]) if ious else np.nan
            other[i] = max((v for p, v in ious if p != r.plane), default=np.nan)
            same[i] = max((v for p, v in ious if p == r.plane), default=np.nan)
    pred["support"] = pd.Series(support)
    pred["iou_other_plane"] = pd.Series(other)
    pred["iou_same_plane"] = pd.Series(same)
    # Files from before p_mean existed fall back to the peak probability.
    p = pred.p_mean if "p_mean" in pred else pred.conf
    pred["confidence"] = p * (0.5 + 0.5 * pred.support.fillna(0))
    return pred


def registered_box(target, sources):
    """Box in `target` (a volumes.csv row) from trusted boxes of other series.

    `sources`: rows with SeriesInstanceUID, plane, a `world` (lo, hi) box and
    `confidence`. The most confident box per plane; intersected across planes
    when they overlap, else the most confident alone.
    """
    best = sources.sort_values("confidence", ascending=False).drop_duplicates("plane")
    boxes = list(best.world)
    lo, hi = np.max([b[0] for b in boxes], 0), np.min([b[1] for b in boxes], 0)
    if (hi <= lo).any():
        lo, hi = boxes[0]
    return world_to_voxel_box(target.SeriesInstanceUID, lo, hi,
                              (target.n_slices, target.rows, target.cols))


def fill_by_registration(boxes, series, trusted, weak=None):
    """Give every (series, class) of the studies in `boxes` a box.

    boxes: per-(series, class) rows with world boxes and `confidence` (from
           `add_confidence`); trusted: boolean mask over `boxes` of the ones that
           may be moved into other series; weak: boolean mask of boxes to replace
           (default: none -- only missing ones are filled).
    Returns the boxes with replacements, plus the filled ones, `source` set to
    "registered" on those.
    """
    weak = pd.Series(False, index=boxes.index) if weak is None else weak
    keep = boxes[~weak].copy()
    have = set(zip(keep.SeriesInstanceUID, keep["class"]))
    src = boxes[trusted]
    rows = []
    for study, g in series[series.StudyInstanceUID.isin(set(src.StudyInstanceUID))].groupby("StudyInstanceUID"):
        s_study = src[src.StudyInstanceUID == study]
        for cls, s_cls in s_study.groupby("class"):
            for t in g.itertuples():
                if (t.SeriesInstanceUID, cls) in have:
                    continue
                s = s_cls[s_cls.SeriesInstanceUID != t.SeriesInstanceUID]
                if s.empty:
                    continue
                box = registered_box(t, s)
                if box is None:
                    continue
                rows.append(dict(SeriesInstanceUID=t.SeriesInstanceUID, StudyInstanceUID=study,
                                 plane=t.Anatomical_Plane, **{"class": cls}, source="registered",
                                 confidence=float(s.confidence.max()), **box))
    filled = pd.DataFrame(rows)
    return pd.concat([keep, filled], ignore_index=True), len(filled)
