"""Patient-space geometry of a series, to compare and move ROI boxes between the series of a study.

All series of a study are positioned in the same patient coordinate system (DICOM LPS, mm), so a box can be mapped
from one series into another (sagittal <-> coronal <-> axial) through each series' geometry. Knee series are routinely
oblique, so the full ImageOrientationPatient is used, not one IPP coordinate per plane.

Training series read their geometry from the per-slice tags saved by `prepare_data.py` (output/metadata_shards);
at test time `infer.pipeline` reads it from the DICOM headers and registers it in `GEOMETRY_OVERRIDE`.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from params import OUTPUT_FOLDER

# Geometry of series that have no metadata shard (test time), keyed by SeriesInstanceUID.
GEOMETRY_OVERRIDE = {}


def _vec(v):
    return np.array(json.loads(v) if isinstance(v, str) else list(v), float)


def geometry_from_slices(iops, ipps, pixel_spacing, slice_thickness=None):
    """-> dict(spacing (x, y, z), direction (9,), origin (3,), n_slices, uniform), from per-slice IOP / IPP lists in
    stored order.

    Index axes: x = column, y = row, z = slice (sorted along the slice normal). Row cosines are IOP[:3] (direction of
    increasing column), column cosines IOP[3:]; the slice axis is their cross product.
    """
    iop = np.asarray(iops[0], float)
    ps = np.asarray(pixel_spacing, float)              # (row spacing, column spacing)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ipp = np.asarray(ipps, float)
    pos = ipp @ normal
    if len(pos) > 1:
        steps = np.diff(pos)
        dz = float(np.median(steps))
        uniform = bool(np.allclose(steps, dz, rtol=0.05, atol=0.05))
    else:
        dz, uniform = float(slice_thickness or 1.0), True
    if dz < 0:                                         # keep a right-handed, positive z step
        normal, dz = -normal, -dz
    return dict(spacing=(float(ps[1]), float(ps[0]), dz),
                direction=tuple(np.column_stack([row_dir, col_dir, normal]).ravel()),
                origin=tuple(ipp[0]), n_slices=len(ipp), uniform=uniform)


def series_geometry(series_uid):
    """`geometry_from_slices` of a training series (metadata shard), or the registered test-time geometry."""
    if series_uid in GEOMETRY_OVERRIDE:
        return GEOMETRY_OVERRIDE[series_uid]
    m = pd.read_pickle(os.path.join(OUTPUT_FOLDER, "metadata_shards", f"{series_uid}.pkl")).sort_values("slice_idx")
    first = m.iloc[0]
    return geometry_from_slices([_vec(first.ImageOrientationPatient)], [_vec(v) for v in m.ImagePositionPatient],
                                _vec(first.PixelSpacing), first.SliceThickness)
