"""infer.seg.consensus with a loop-light roi.consensus.add_confidence (same values, no per-row pandas indexing)."""

import numpy as np
import pandas as pd

from roi.boxes import iou
from roi.consensus import BOX, _world, fill_by_registration


def add_confidence(pred):
    pred = pred[pred.z0.notna()].copy().reset_index(drop=True)
    pred["world"] = _world(pred)
    n = len(pred)
    support, other, same = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    world, plane = list(pred.world), list(pred.plane)
    for idx in pred.groupby(["StudyInstanceUID", "class"]).indices.values():
        for i in idx:
            ious = [(plane[j], iou(world[i], world[j])) for j in idx if j != i]
            if ious:
                support[i] = np.mean([v > 0 for _, v in ious])
            other[i] = max((v for p, v in ious if p != plane[i]), default=np.nan)
            same[i] = max((v for p, v in ious if p == plane[i]), default=np.nan)
    pred["support"], pred["iou_other_plane"], pred["iou_same_plane"] = support, other, same
    p = pred.p_mean if "p_mean" in pred else pred.conf
    pred["confidence"] = p * (0.5 + 0.5 * pred.support.fillna(0))
    return pred


def consensus(pred, rows):
    """infer.seg.consensus (knee = study + predicted side)."""
    knee = lambda d: d.StudyInstanceUID + "|" + d.side  # noqa: E731
    pred["study"] = pred.StudyInstanceUID
    pred["StudyInstanceUID"] = knee(pred)
    pred = add_confidence(pred)
    weak = pred.conf < 0.5
    knees = rows.assign(StudyInstanceUID=knee(rows), Anatomical_Plane=rows.plane)
    filled, n_filled = fill_by_registration(pred.assign(source="unet"), knees, ~weak, weak)
    filled["StudyInstanceUID"] = filled.StudyInstanceUID.str.split("|").str[0]
    filled["side"] = filled.SeriesInstanceUID.map(rows.set_index("SeriesInstanceUID").side)
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "plane", "side", "class"] + BOX + \
        ["conf", "p_mean", "support", "iou_other_plane", "iou_same_plane", "confidence", "source"]
    filled = filled.reindex(columns=cols).sort_values(["StudyInstanceUID", "SeriesInstanceUID", "class"])
    return filled, int(weak.sum()), n_filled
