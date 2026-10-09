"""Put the series of a study in a common world space, to move segmentations
between views. Used by `notebooks/05_registration.ipynb`.

The use case: nnU-Net (OAI-ZIB, trained on 3D sagittal knees) is reasonable on
sagittal series, weaker on coronal and poor on axial ones. An axial segmentation
can instead be obtained by resampling the sagittal (and coronal) predictions onto
the axial voxel grid, since all series of a study are positioned in the same
patient coordinate system (DICOM LPS, mm).

Geometry comes from the per-slice DICOM tags saved by `01_preparation.ipynb`
(ImagePositionPatient, ImageOrientationPatient, PixelSpacing) and is used in full:
knee series are routinely oblique, so the axis-aligned shortcut of the lumbar
spine notebook (one IPP coordinate per plane) does not hold.

Caveats:
  - every series carries its own FrameOfReferenceUID (most likely remapped by the
    anonymisation), so a shared frame cannot be checked from tags: alignment is
    checked on the images, and `rigid_refine` corrects patient motion;
  - a sagittal series has 3-4 mm gaps along left-right, which becomes an in-plane
    axis of an axial slice. Labels are resampled as one-hot probabilities with
    linear interpolation (smooth, not blocky), and fusing sagittal with coronal
    (fine left-right sampling) fills in what one view lacks.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "output")


def _vec(v):
    return np.array(json.loads(v) if isinstance(v, str) else list(v), float)


# Geometry of series that have no metadata shard (test time): `infer.prep` reads
# it from the DICOM headers and registers it here, keyed by SeriesInstanceUID.
GEOMETRY_OVERRIDE = {}


def geometry_from_slices(iops, ipps, pixel_spacing, slice_thickness=None):
    """series_geometry's dict from per-slice IOP / IPP lists, already in stored order."""
    iop = np.asarray(iops[0], float)
    ps = np.asarray(pixel_spacing, float)
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
    if dz < 0:
        normal, dz = -normal, -dz
    return dict(spacing=(float(ps[1]), float(ps[0]), dz),
                direction=tuple(np.column_stack([row_dir, col_dir, normal]).ravel()),
                origin=tuple(ipp[0]), n_slices=len(ipp), uniform=uniform)


def series_geometry(series_uid):
    """-> dict(spacing (x, y, z), direction (9,), origin (3,), n_slices, uniform).

    Index axes: x = column, y = row, z = slice (as stored in series_npy, sorted
    along the slice normal). Row cosines are IOP[:3] (direction of increasing
    column), column cosines IOP[3:]; the slice axis is their cross product, the
    direction along which `01_preparation` sorted the slices.
    """
    if series_uid in GEOMETRY_OVERRIDE:
        return GEOMETRY_OVERRIDE[series_uid]
    m = pd.read_pickle(os.path.join(OUT, "metadata_shards", f"{series_uid}.pkl")).sort_values("slice_idx")
    first = m.iloc[0]
    iop = _vec(first.ImageOrientationPatient)
    ps = _vec(first.PixelSpacing)                      # (row spacing, column spacing)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    ipp = np.stack([_vec(v) for v in m.ImagePositionPatient])
    pos = ipp @ normal
    if len(pos) > 1:
        steps = np.diff(pos)
        dz = float(np.median(steps))
        uniform = bool(np.allclose(steps, dz, rtol=0.05, atol=0.05))
    else:
        dz, uniform = float(first.SliceThickness), True
    if dz < 0:                                         # keep a right-handed, positive z step
        normal, dz = -normal, -dz
    return dict(spacing=(float(ps[1]), float(ps[0]), dz),
                direction=tuple(np.column_stack([row_dir, col_dir, normal]).ravel()),
                origin=tuple(ipp[0]), n_slices=len(m), uniform=uniform)


def series_image(series_uid, array=None):
    """The raw series (output/series_npy) as a float32 sitk image with its world geometry."""
    if array is None:
        array = np.load(os.path.join(OUT, "series_npy", f"{series_uid}.npy"))
    g = series_geometry(series_uid)
    img = sitk.GetImageFromArray(np.asarray(array, np.float32))
    img.SetSpacing(g["spacing"])
    img.SetDirection(g["direction"])
    img.SetOrigin(g["origin"])
    return img


def with_geometry(array, reference):
    """Wrap a (z, y, x) array in the geometry of `reference`."""
    img = sitk.GetImageFromArray(array)
    img.CopyInformation(reference)
    return img


def resample(moving, reference, transform=None, interpolator=sitk.sitkLinear, default=0.0):
    tx = transform if transform is not None else sitk.Transform()
    return sitk.Resample(moving, reference, tx, interpolator, default, moving.GetPixelID())


def label_probabilities(seg, reference, n_labels, transform=None):
    """One-hot labels of `seg` resampled onto `reference`.

    -> (probs (n_labels + 1, z, y, x) float32, coverage (z, y, x) float32).
    Linear interpolation of the one-hot maps gives soft labels between source
    slices. `coverage` is 1 inside the source volume and 0 outside, so fusion
    can tell "background" from "not seen by this view".
    """
    seg_np = sitk.GetArrayFromImage(seg)
    probs = []
    for lab in range(n_labels + 1):
        oh = with_geometry((seg_np == lab).astype(np.float32), seg)
        probs.append(sitk.GetArrayFromImage(resample(oh, reference, transform)))
    ones = with_geometry(np.ones(seg_np.shape, np.float32), seg)
    coverage = sitk.GetArrayFromImage(resample(ones, reference, transform))
    return np.stack(probs).astype(np.float32), coverage.astype(np.float32)


def fuse(prob_list, coverage_list, weights=None, min_coverage=0.5):
    """Weighted soft vote of several views -> label map (z, y, x) uint8.

    Each view only votes where it covers the voxel. Voxels no view covers are
    background.
    """
    weights = weights or [1.0] * len(prob_list)
    acc = np.zeros_like(prob_list[0])
    for p, c, w in zip(prob_list, coverage_list, weights):
        acc += w * p * (c >= min_coverage)[None]
    lab = acc.argmax(0).astype(np.uint8)
    lab[acc.sum(0) == 0] = 0
    return lab


def rigid_refine(fixed, moving, iterations=200, sampling=0.2, seed=0):
    """Rigid (Euler 3D) registration of `moving` onto `fixed`, from the identity.

    The world coordinates are already a good start, so this only corrects patient
    motion between acquisitions. Mattes mutual information, as the series have
    different contrasts (PD fat-sat, T1, T2). Returns (transform, metric_before,
    metric_after); lower metric is better. The transform maps fixed-space points
    to moving space, i.e. what `resample(moving, fixed, transform)` expects.
    """
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-3, numberOfIterations=iterations, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    tx = sitk.Euler3DTransform()
    centre = fixed.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in fixed.GetSize()])
    tx.SetCenter(centre)
    reg.SetInitialTransform(tx, inPlace=False)
    f, m = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
    before = _metric(f, m, sitk.Euler3DTransform(tx), sampling, seed)
    out = reg.Execute(f, m)
    return out, before, _metric(f, m, out, sampling, seed)


def _metric(fixed, moving, transform, sampling, seed):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(transform)
    return reg.MetricEvaluate(fixed, moving)


def transform_summary(tx):
    """(max rotation in degrees, translation norm in mm) of a rigid transform."""
    t = sitk.CompositeTransform(tx).GetNthTransform(0) if tx.GetName() == "CompositeTransform" else tx
    t = sitk.Euler3DTransform(t)
    angles = np.degrees([t.GetAngleX(), t.GetAngleY(), t.GetAngleZ()])
    return float(np.abs(angles).max()), float(np.linalg.norm(t.GetTranslation()))


def slice_lines(img, k, other):
    """Where each slice of `other` cuts slice k of `img`, in img pixel coordinates.

    -> list of (slice index in other, (x0, y0), (x1, y1)), one segment per slice
    of `other` whose plane crosses the slice. The knee version of the lumbar
    notebook's "axial levels on the sagittal": exact for oblique series.
    """
    d = np.array(img.GetDirection()).reshape(3, 3)
    sx, sy, sz = img.GetSpacing()
    o = np.array(img.GetOrigin()) + k * sz * d[:, 2]
    u, v = sx * d[:, 0], sy * d[:, 1]                  # world step per column / per row
    do = np.array(other.GetDirection()).reshape(3, 3)
    n2 = do[:, 2]
    o2 = np.array(other.GetOrigin())
    W, H = img.GetSize()[0], img.GetSize()[1]
    a, b = u @ n2, v @ n2                              # a*x + b*y + c0 = position along n2
    c0 = (o - o2) @ n2
    lines = []
    for k2 in range(other.GetSize()[2]):
        c = k2 * other.GetSpacing()[2] - c0
        pts = []
        if abs(b) > 1e-9:
            for x in (0, W - 1):
                y = (c - a * x) / b
                if 0 <= y <= H - 1:
                    pts.append((x, y))
        if abs(a) > 1e-9:
            for y in (0, H - 1):
                x = (c - b * y) / a
                if 0 <= x <= W - 1:
                    pts.append((x, y))
        if len(pts) >= 2:
            lines.append((k2, pts[0], pts[-1]))
    return lines


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")
