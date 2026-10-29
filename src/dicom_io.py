"""DICOM reading rules shared by the training data preparation (`prepare_data.py`) and inference (`infer.pipeline`)."""

import numpy as np

PLANES = ["Sagittal", "Coronal", "Axial"]


def orientation(iop):
    """Plane of a slice from its ImageOrientationPatient: the dominant axis of the slice normal."""
    n = np.abs(np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float)))
    return PLANES[int(np.argmax(n))]


def slice_order(dss):
    """-> (order, majority plane or None): slices sorted by their position along the slice normal of the majority
    orientation (slices with an outlier orientation do not influence the normal), ties broken by InstanceNumber.
    Without usable geometry: InstanceNumber order."""
    inst = np.array([int(d.get("InstanceNumber", 0) or 0) for d in dss])
    try:
        iops = [np.array(d.ImageOrientationPatient, float) for d in dss]
        orients = [orientation(i) for i in iops]
        major = max(set(orients), key=orients.count)
        normals = np.array([np.cross(i[:3], i[3:]) for i in iops])
        normal = normals[[o == major for o in orients]].mean(0)
        normal /= np.linalg.norm(normal)
        pos = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in dss])
        return np.lexsort((inst, pos)), major
    except Exception:  # noqa: BLE001 -- no usable geometry
        return np.argsort(inst, kind="stable"), None


def to_monochrome2(vol, dss):
    """Slices stored as MONOCHROME1 (minimum value = white) inverted to the MONOCHROME2 convention every model was
    trained on: v -> min + max - v over the volume, in its dtype. All training series are MONOCHROME2; this guards
    the hidden test set."""
    inv = np.array([str(d.get("PhotometricInterpretation", "")).strip().upper() == "MONOCHROME1" for d in dss])
    if not inv.any():
        return vol
    lo, hi = int(vol.min()), int(vol.max())
    out = vol.copy()
    out[inv] = (lo + hi - vol[inv].astype(np.int64)).astype(vol.dtype)
    return out


def pixel_spacing(ds):
    """(row, column) spacing in mm: PixelSpacing, else ImagerPixelSpacing, else square pixels over a 160 mm field of
    view on the longer side (the training median in every plane). Spacing only scales the patient-space geometry the
    cross-plane box transfer uses, so a missing tag must not drop the series."""
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        try:
            v = [float(x) for x in ds.get(tag)]
            if len(v) == 2 and all(np.isfinite(v)) and min(v) > 0:
                return v
        except Exception:  # noqa: BLE001 -- absent / malformed
            pass
    return [160.0 / max(int(ds.Rows), int(ds.Columns))] * 2
