"""Boxes from 3D maps (attention or U-Net output) and how much two views agree on them.

A map here is a (F, G, G) volume over the frames of a series, in-plane on the
slice zero-padded to a centred square, as written by `roi.extract`. Two ways to
turn it into a box:

  - "zrun": the contiguous frames around the peak of the z weights (>= z_rel x
    peak), their in-plane maps averaged; in-plane, the main connected component
    of the smallest region holding `mass` of the attention. What
    `attention_boxes.py` did.
  - "cc3d": the 3D attention itself (z weight x in-plane map); the smallest set of
    voxels holding `mass` of it, and its 26-connected component with the most
    attention. One box in 3D directly, no separate z and xy decisions.

Localised = compact: in-plane box area / slice area <= max_area and frames
spanned / frames <= max_z_frac. Otherwise no box.

Cross-view agreement. A study's sagittal and coronal series are independent
acquisitions; the boxes of a class in the two, mapped to patient coordinates
(the 8 corners, then the axis-aligned box around them, in mm), should overlap.
With no ground-truth boxes, agreement is the measure of box quality.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy import ndimage as ndi


def smallest_mass_set(a, mass):
    flat = a.ravel() / max(a.sum(), 1e-12)
    order = np.argsort(-flat)
    keep = np.zeros(flat.size, bool)
    keep[order[:np.searchsorted(np.cumsum(flat[order]), mass) + 1]] = True
    return keep.reshape(a.shape)


def _main_component(region, weights, structure):
    lab, n = ndi.label(region, structure=structure)
    if n == 0:
        return None, 0.0
    comp_mass = ndi.sum(weights, lab, index=np.arange(1, n + 1))
    main = int(np.argmax(comp_mass)) + 1
    return lab == main, float(comp_mass[main - 1] / max(comp_mass.sum(), 1e-12))


def grid_box(xy, z, mode="cc3d", mass=0.5, z_rel=0.5):
    """-> dict(f0, f1, gy0, gy1, gx0, gx1, purity) on the frame / grid axes."""
    if mode == "zrun":
        peak = int(np.argmax(z))
        keep = z >= z_rel * z[peak]
        lo = hi = peak
        while lo > 0 and keep[lo - 1]:
            lo -= 1
        while hi < len(z) - 1 and keep[hi + 1]:
            hi += 1
        w = z[lo:hi + 1] / z[lo:hi + 1].sum()
        avg = np.tensordot(w, xy[lo:hi + 1].astype(np.float32), axes=1)
        comp, purity = _main_component(smallest_mass_set(avg, mass), avg, np.ones((3, 3)))
        ys, xs = np.nonzero(comp)
        return dict(f0=lo, f1=hi, gy0=ys.min(), gy1=ys.max(), gx0=xs.min(), gx1=xs.max(),
                    purity=purity)
    vol = xy.astype(np.float32) * z[:, None, None]
    comp, purity = _main_component(smallest_mass_set(vol, mass), vol, np.ones((3, 3, 3)))
    fs, ys, xs = np.nonzero(comp)
    return dict(f0=fs.min(), f1=fs.max(), gy0=ys.min(), gy1=ys.max(), gx0=xs.min(), gx1=xs.max(),
                purity=purity)


def to_voxels(b, slices, rows, cols, grid):
    """Grid box -> original voxel indices (slices z0..z1, rows y0..y1, cols x0..x1), inclusive."""
    s = max(rows, cols)
    oy, ox = (s - rows) // 2, (s - cols) // 2
    cell = s / grid
    y0 = int(np.floor(b["gy0"] * cell)) - oy
    y1 = int(np.ceil((b["gy1"] + 1) * cell)) - 1 - oy
    x0 = int(np.floor(b["gx0"] * cell)) - ox
    x1 = int(np.ceil((b["gx1"] + 1) * cell)) - 1 - ox
    return dict(z0=int(slices[b["f0"]]), z1=int(slices[b["f1"]]),
                y0=max(y0, 0), y1=min(y1, rows - 1), x0=max(x0, 0), x1=min(x1, cols - 1))


def localisation(b, n_frames, grid, rows, cols):
    """Area of the in-plane box as a share of the (unpadded) slice, and z extent share."""
    s = max(rows, cols)
    area = ((b["gy1"] - b["gy0"] + 1) * (b["gx1"] - b["gx0"] + 1) * (s / grid) ** 2) / (rows * cols)
    return dict(area=float(min(area, 1.0)), z_frac=float((b["f1"] - b["f0"] + 1) / n_frames))


# ------------------------------------------------------------------- agreement


@functools.lru_cache(maxsize=None)
def _geometry(series_uid):
    from knee_registration import series_geometry

    g = series_geometry(series_uid)
    d = np.array(g["direction"]).reshape(3, 3)
    return np.array(g["origin"]), d * np.array(g["spacing"])[None, :]


def world_box(series_uid, z0, z1, y0, y1, x0, x1):
    """Axis-aligned box in patient coordinates (mm) around a voxel box's 8 corners."""
    origin, m = _geometry(series_uid)
    corners = np.array([(x, y, z) for x in (x0 - 0.5, x1 + 0.5) for y in (y0 - 0.5, y1 + 0.5)
                        for z in (z0 - 0.5, z1 + 0.5)])
    pts = origin + corners @ m.T
    return pts.min(0), pts.max(0)


def iou(a, b):
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    inter = np.prod(np.clip(hi - lo, 0, None))
    union = np.prod(a[1] - a[0]) + np.prod(b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def cross_view(boxes):
    """Per (study, class): agreement of the sagittal and coronal boxes.

    `boxes`: DataFrame with StudyInstanceUID, SeriesInstanceUID, plane, class,
    z0..x1 (localised boxes only). Every sagittal x coronal pair is scored.
    Returns one row per pair: study, class, iou, overlap (iou > 0).
    """
    import pandas as pd

    rows = []
    for (study, cls), g in boxes.groupby(["StudyInstanceUID", "class"]):
        sag, cor = g[g.plane == "Sagittal"], g[g.plane == "Coronal"]
        for a in sag.itertuples():
            wa = world_box(a.SeriesInstanceUID, a.z0, a.z1, a.y0, a.y1, a.x0, a.x1)
            for b in cor.itertuples():
                wb = world_box(b.SeriesInstanceUID, b.z0, b.z1, b.y0, b.y1, b.x0, b.x1)
                v = iou(wa, wb)
                rows.append(dict(StudyInstanceUID=study, cls=cls, iou=v, overlap=v > 0))
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "cls", "iou", "overlap"])
