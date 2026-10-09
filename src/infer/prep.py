"""DICOM series -> the volumes the side classifier and the ROI U-Net read, run from src/.

    python -m infer.prep --data_dir <dir with test_series/ and test_series.csv> --out_dir /tmp/knee/

Per series, in a pool of processes:
  1. read every slice, order them as `notebooks/01_preparation.ipynb` did for
     training (position along the slice normal of the majority orientation,
     InstanceNumber to break ties), stack the pixel arrays (raw, no rescale);
  2. `prepare_cache.preprocess_volume` (per-series percentile normalisation,
     centred square padding, 384 px, at most 64 slices) then
     `roi.prepare.cache_to_volume` (32 x 128 x 128): the exact training path;
  3. read the geometry (`knee_registration.geometry_from_slices`) and the
     Laterality tag;
  4. left-right convention: in every training series the left-right axis points
     the same way (sagittal slices towards the patient's right, coronal / axial
     columns towards the left). A test series that does not is flipped along that
     axis, and `conv_flip` records it so its boxes are mapped back.

Writes to --out_dir:
  volumes.npy     (N, 32, 128, 128) uint8, in the training convention
  series.csv      row, SeriesInstanceUID, StudyInstanceUID, plane, n_slices,
                  n_cache, rows, cols, conv_flip, dicom_laterality, status
  instances.csv   SeriesInstanceUID, slice_idx, SOPInstanceUID: which file each
                  slice index of the boxes refers to
  geometry.json   per series, the geometry for patient-space comparisons
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

DEPTH, SIZE = 32, 128


def _orient(iop):
    n = np.abs(np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float)))
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(n))]


def slice_order(dss):
    """As in 01_preparation: sort along the majority orientation's slice normal."""
    inst = np.array([int(d.get("InstanceNumber", 0) or 0) for d in dss])
    try:
        iops = [np.array(d.ImageOrientationPatient, float) for d in dss]
        orients = [_orient(i) for i in iops]
        major = max(set(orients), key=orients.count)
        normals = np.array([np.cross(i[:3], i[3:]) for i in iops])
        normal = normals[[o == major for o in orients]].mean(0)
        normal /= np.linalg.norm(normal)
        pos = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in dss])
        return np.lexsort((inst, pos)), major
    except Exception:  # noqa: BLE001 -- no usable geometry: InstanceNumber order
        return np.argsort(inst, kind="stable"), None


def read_series(folder):
    import pydicom

    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    dss = [pydicom.dcmread(f) for f in files]
    order, major = slice_order(dss)
    dss = [dss[i] for i in order]
    arrs = [d.pixel_array for d in dss]
    arrs = [a[0] if a.ndim == 3 and a.shape[0] == 1 else a for a in arrs]
    shapes = [(a.shape, a.dtype) for a in arrs]
    if len(set(shapes)) > 1:              # keep the majority shape rather than fail
        keep = max(set(shapes), key=shapes.count)
        dss = [d for d, s in zip(dss, shapes) if s == keep]
        arrs = [a for a, s in zip(arrs, shapes) if s == keep]
    vol = np.stack(arrs)
    tag = next((str(d.get("Laterality", "")).upper()[:1] for d in dss if d.get("Laterality", "")), "")
    return vol, dss, major, tag


def process(job):
    import torch

    from knee_registration import geometry_from_slices
    from prepare_cache import preprocess_volume
    from roi.prepare import cache_to_volume

    torch.set_num_threads(1)
    study, series, folder, plane = job
    try:
        vol, dss, major, tag = read_series(folder)
        plane = plane if isinstance(plane, str) and plane else (major or "Sagittal")
        geom = geometry_from_slices([d.ImageOrientationPatient for d in dss],
                                    [d.ImagePositionPatient for d in dss],
                                    dss[0].PixelSpacing, dss[0].get("SliceThickness"))
        cache = preprocess_volume(vol)
        volume = cache_to_volume(cache)
        d = np.array(geom["direction"]).reshape(3, 3)
        # Training convention: sagittal slices towards patient right (-x), columns towards left (+x).
        conv_flip = bool(d[0, 2] > 0) if plane == "Sagittal" else bool(d[0, 0] < 0)
        if conv_flip:
            volume = np.ascontiguousarray(np.flip(volume, 0 if plane == "Sagittal" else 2))
        meta = dict(SeriesInstanceUID=series, StudyInstanceUID=study, plane=plane,
                    n_slices=int(vol.shape[0]), n_cache=int(cache.shape[0]), rows=int(vol.shape[1]),
                    cols=int(vol.shape[2]), conv_flip=conv_flip, dicom_laterality=tag, status="ok")
        sops = [str(x.SOPInstanceUID) for x in dss]
        return volume, meta, geom, sops
    except Exception as e:  # noqa: BLE001 -- one bad series must not stop the run
        return None, dict(SeriesInstanceUID=series, StudyInstanceUID=study, plane=plane,
                          status=f"failed: {e!r}"[:300]), None, []


def run(data_dir, out_dir, workers=None, limit=None, csv_name="test_series.csv", series_dir="test_series"):
    os.makedirs(out_dir, exist_ok=True)
    meta_csv = os.path.join(data_dir, csv_name)
    if os.path.exists(meta_csv):
        df = pd.read_csv(meta_csv)
    else:                                  # no series csv: walk the folders, plane from geometry
        rows = [(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                for p in sorted(glob.glob(os.path.join(data_dir, series_dir, "*", "*")))]
        df = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID"]).assign(Anatomical_Plane="")
    if limit:
        df = df[df.StudyInstanceUID.isin(df.StudyInstanceUID.drop_duplicates()[:limit])]
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
             os.path.join(data_dir, series_dir, r.StudyInstanceUID, r.SeriesInstanceUID),
             getattr(r, "Anatomical_Plane", "")) for r in df.itertuples()]
    vols = np.lib.format.open_memmap(os.path.join(out_dir, "volumes.npy"), mode="w+", dtype=np.uint8,
                                     shape=(len(jobs), DEPTH, SIZE, SIZE))
    metas, geoms, inst = [], {}, []
    t0 = time.time()
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for i, (v, meta, geom, sops) in enumerate(ex.map(process, jobs, chunksize=1)):
            meta["row"] = i
            metas.append(meta)
            if v is not None:
                vols[i] = v
                geoms[meta["SeriesInstanceUID"]] = geom
                inst += [(meta["SeriesInstanceUID"], k, s) for k, s in enumerate(sops)]
    vols.flush()
    series = pd.DataFrame(metas)
    series.to_csv(os.path.join(out_dir, "series.csv"), index=False)
    pd.DataFrame(inst, columns=["SeriesInstanceUID", "slice_idx", "SOPInstanceUID"]).to_csv(
        os.path.join(out_dir, "instances.csv"), index=False)
    with open(os.path.join(out_dir, "geometry.json"), "w") as f:
        json.dump(geoms, f)
    n_ok = int((series.status == "ok").sum())
    print(f"-> {n_ok}/{len(series)} series of {series.StudyInstanceUID.nunique()} studies "
          f"in {time.time() - t0:.0f}s ({(time.time() - t0) / max(len(series), 1):.2f} s/series), "
          f"{int(series.get('conv_flip', pd.Series(dtype=bool)).fillna(False).sum())} flipped to the "
          f"training left-right convention -> {out_dir}")
    for r in series[series.status != "ok"].itertuples():
        print("  FAILED", r.SeriesInstanceUID, r.status)
    return series


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="first N studies (debugging)")
    p.add_argument("--csv_name", default="test_series.csv")
    p.add_argument("--series_dir", default="test_series")
    a = p.parse_args()
    run(a.data_dir, a.out_dir, a.workers, a.limit, a.csv_name, a.series_dir)


if __name__ == "__main__":
    main()
