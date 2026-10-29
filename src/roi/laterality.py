"""Medial / lateral sanity checks of ROI boxes, from the knee's side (`roi.evaluate` reports them).

Cross-view agreement cannot catch a medial / lateral swap made the same way in
both views. Laterality can: in patient coordinates (DICOM LPS, +x towards the
patient's left) the medial side of a right knee faces +x, that of a left knee -x.
So in a series with both boxes, the medial one must lie on the correct side of
the lateral one. The side of each series is the side classifier's (`side_cls.py`).

Per pair (menisci, tibiofemoral OA compartments): the share of series where the
medial box is on the medial side. 0.5 would be a coin flip. Also reported: the
share of series whose meniscus box lies in its own compartment's box
(`pair_consistency`). The label merges (`roi.labels merge`) drop the boxes that
fail either check.
"""

import functools
import os

import numpy as np
import pandas as pd

from params import OUTPUT_FOLDER
from roi.boxes import world_box

PAIRS = [("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")]


@functools.lru_cache(maxsize=None)
def series_side():
    """SeriesInstanceUID -> 'R' / 'L' from the side classifier (output/series_laterality.csv).

    Per series, since a study can hold both knees, and from the model rather
    than the DICOM tag, which is wrong on ~1-2% of tagged series (`side_cls`).
    """
    lat = pd.read_csv(os.path.join(OUTPUT_FOLDER, "series_laterality.csv"))
    return lat.dropna(subset=["side_model"]).set_index("SeriesInstanceUID").side_model.to_dict()


def study_side():
    """StudyInstanceUID -> majority side of its series (for study-level priors)."""
    lat = pd.read_csv(os.path.join(OUTPUT_FOLDER, "series_laterality.csv")).dropna(subset=["side_model"])
    return lat.groupby("StudyInstanceUID").side_model.agg(lambda s: s.value_counts().index[0]).to_dict()


def check(boxes):
    """Share of series whose medial box lies on the medial side, per pair; side from `series_side`."""
    side = series_side()
    boxes = boxes[boxes.SeriesInstanceUID.isin(side) & boxes.z0.notna()]
    out = {}
    for med, lat in PAIRS:
        b = boxes[boxes["class"].isin([med, lat])]
        ok = []
        for uid, g in b.groupby("SeriesInstanceUID"):
            if set(g["class"]) != {med, lat}:
                continue
            cx = {}
            for _, r in g.iterrows():
                lo, hi = world_box(uid, int(r.z0), int(r.z1), int(r.y0), int(r.y1), int(r.x0), int(r.x1))
                cx[r["class"]] = (lo[0] + hi[0]) / 2
            sign = 1 if side[uid] == "R" else -1
            ok.append(sign * (cx[med] - cx[lat]) > 0)
        out[med.split()[1]] = (float(np.mean(ok)) if ok else np.nan, len(ok))
    return out


def wrong_side(labels, side=None):
    """Boolean mask of the medial / lateral labels that sit on the wrong side.

    `side`: SeriesInstanceUID -> side, default `series_side()`.

    Judged within a series, against the series' box of the paired class (from
    `labels` itself): both boxes of a wrongly ordered pair are flagged, as there
    is no telling which one is off.
    """
    side = series_side() if side is None else side
    bad = pd.Series(False, index=labels.index)
    for med, lat in PAIRS:
        b = labels[labels["class"].isin([med, lat]) & labels.SeriesInstanceUID.isin(side)]
        for uid, g in b.groupby("SeriesInstanceUID"):
            if set(g["class"]) != {med, lat}:
                continue
            cx = {}
            for i, r in g.iterrows():
                lo, hi = world_box(uid, int(r.z0), int(r.z1), int(r.y0), int(r.y1), int(r.x0), int(r.x1))
                cx[r["class"]] = (lo[0] + hi[0]) / 2
            sign = 1 if side[uid] == "R" else -1
            if sign * (cx[med] - cx[lat]) <= 0:
                bad[g.index] = True
    return bad




def pair_consistency(boxes):
    """Share of series whose meniscus box centre lies inside the same-side compartment box.

    The medial meniscus sits in the medial tibiofemoral compartment, the lateral
    one in the lateral compartment, so in a series with both boxes of a side the
    meniscus centre should fall in the compartment box (series voxel space).
    Returns {side: (share, n)} and the index of the inconsistent rows.
    """
    out, bad = {}, []
    b = boxes[boxes.z0.notna()]
    for side in ("Medial", "Lateral"):
        men, comp = f"{side} Meniscus", f"{side} OA"
        ok = []
        for _, g in b[b["class"].isin([men, comp])].groupby("SeriesInstanceUID"):
            if set(g["class"]) != {men, comp}:
                continue
            m, c = g[g["class"] == men].iloc[0], g[g["class"] == comp].iloc[0]
            inside = all(c[f"{k}0"] <= (m[f"{k}0"] + m[f"{k}1"]) / 2 <= c[f"{k}1"] for k in "zyx")
            ok.append(inside)
            if not inside:
                bad.extend(g.index)
        out[side] = (float(np.mean(ok)) if ok else np.nan, len(ok))
    return out, bad

