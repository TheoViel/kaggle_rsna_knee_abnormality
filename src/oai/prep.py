"""OAI baseline series (input/oai/images) -> our cache / volumes / metadata, run from src/.

    python -m oai.prep --workers 64

Per series archive (tar.gz of DICOM slices), in a pool of processes, exactly the test-time path of
`infer.prep` (slice order, `prepare_cache.preprocess_volume` -> 384 px cache, `cache_to_volume` ->
32 x 128 x 128 volume, left-right training convention via `conv_flip`), with one adaptation:
thin-slice 3D series (the 0.7 mm sagittal DESS and its coronal / axial reformats) are averaged into
~3 mm slabs first, so that their slice spacing matches the clinical series our crops are built on
(8 strided slices then cover the same anatomy).

Writes to output/oai/:
  cache_384/<SeriesInstanceUID>.npy    the 384 px uint8 cache (stored orientation)
  volumes.npy                          (N, 32, 128, 128) uint8, training convention (U-Net input)
  series.csv                           row, SeriesInstanceUID, StudyInstanceUID (OAI_<id>_<side>), plane,
                                       seq, side, Fluid_Sensitive, n_slices, n_cache, rows, cols, conv_flip,
                                       slab, status
  geometry.json                        per series, for registration
"""

import argparse
import io
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from oai.labels import MONTHS, knee_uid, visit_dir
from params import INPUT_FOLDER, OUTPUT_FOLDER

IMG_ROOT = os.path.join(INPUT_FOLDER, "oai", "images", "baseline", "image03")
OAI_DIR = os.path.join(OUTPUT_FOLDER, "oai")
PLANE = {"SAG_IW_TSE": "Sagittal", "COR_IW_TSE": "Coronal", "SAG_3D_DESS": "Sagittal", "COR_MPR": "Coronal", "AX_MPR": "Axial"}
FLUID = {"SAG_IW_TSE": 1, "COR_IW_TSE": 0, "SAG_3D_DESS": 1, "COR_MPR": 1, "AX_MPR": 1}   # IW TSE sag is fat-suppressed
TARGET_MM = 3.0


def read_tar(path):
    import pydicom

    with tarfile.open(path) as t:
        dss = [pydicom.dcmread(io.BytesIO(t.extractfile(m).read())) for m in t.getmembers() if m.isfile()]
    return [d for d in dss if "PixelData" in d]


def slab_average(vol, dss, spacing):
    """Average consecutive slices into ~TARGET_MM slabs (thin 3D series). Returns vol, representative slices."""
    k = int(round(TARGET_MM / spacing))
    if k <= 1:
        return vol, dss, 1
    n = len(vol) // k
    vol = vol[:n * k].reshape(n, k, *vol.shape[1:]).mean(1)
    return vol, [dss[i * k + k // 2] for i in range(n)], k


def process(job):
    import torch

    from infer.prep import slice_order
    from knee_registration import geometry_from_slices
    from prepare_cache import preprocess_volume
    from roi.prepare import cache_to_volume

    torch.set_num_threads(1)
    path, oai_id, side, seq, cache_dir, visit = job
    meta = dict(StudyInstanceUID=knee_uid(oai_id, side, visit), seq=seq, side="R" if side == "RIGHT" else "L",
                plane=PLANE[seq], Fluid_Sensitive=FLUID[seq])
    try:
        dss = read_tar(path)
        order, _ = slice_order(dss)
        dss = [dss[i] for i in order]
        shapes = [d.pixel_array.shape for d in dss]
        keep = max(set(shapes), key=shapes.count)
        dss = [d for d, s in zip(dss, shapes) if s == keep]
        vol = np.stack([d.pixel_array for d in dss]).astype(np.float32)
        pos = np.array([np.dot(np.array(d.ImagePositionPatient, float),
                               np.cross(np.array(d.ImageOrientationPatient[:3], float), np.array(d.ImageOrientationPatient[3:], float)))
                        for d in dss])
        spacing = float(np.median(np.abs(np.diff(pos)))) if len(pos) > 1 else 3.0
        vol, dss, k = slab_average(vol, dss, spacing)
        uid = str(dss[0].SeriesInstanceUID)
        geom = geometry_from_slices([d.ImageOrientationPatient for d in dss], [d.ImagePositionPatient for d in dss],
                                    dss[0].PixelSpacing, dss[0].get("SliceThickness"))
        cache = preprocess_volume(vol)
        np.save(os.path.join(cache_dir, f"{uid}.npy"), cache)
        volume = cache_to_volume(cache)
        d = np.array(geom["direction"]).reshape(3, 3)
        conv_flip = bool(d[0, 2] > 0) if meta["plane"] == "Sagittal" else bool(d[0, 0] < 0)
        if conv_flip:
            volume = np.ascontiguousarray(np.flip(volume, 0 if meta["plane"] == "Sagittal" else 2))
        meta.update(SeriesInstanceUID=uid, n_slices=int(vol.shape[0]), n_cache=int(cache.shape[0]), rows=int(vol.shape[1]),
                    cols=int(vol.shape[2]), conv_flip=conv_flip, slab=k, spacing=spacing, status="ok")
        return volume, meta, geom
    except Exception as e:  # noqa: BLE001
        meta.update(SeriesInstanceUID=f"failed_{os.path.basename(path)}", status=f"failed: {e!r}"[:200])
        return None, meta, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--visit", default="00", choices=["00", "01", "03"], help="MOAKS visit: baseline, 12 or 24 months")
    a = p.parse_args()
    out_dir = visit_dir(a.visit)
    img_root = IMG_ROOT if a.visit == "00" else os.path.join(INPUT_FOLDER, "oai", "images", f"m{MONTHS[a.visit]}", "image03")
    idx = pd.read_csv(os.path.join(INPUT_FOLDER, "oai", "images",
                                   "baseline_moaks_series.csv" if a.visit == "00" else f"m{MONTHS[a.visit]}_moaks_series.csv"))
    if a.limit:
        idx = idx.head(a.limit)
    cache_dir = os.path.join(out_dir, "cache_384")
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(os.path.join(img_root, r.image_file.split("submission_60350/")[1]), r.src_subject_id, r.side, r.seq, cache_dir, a.visit)
            for r in idx.itertuples()]
    vols = np.lib.format.open_memmap(os.path.join(out_dir, "volumes.npy"), mode="w+", dtype=np.uint8, shape=(len(jobs), 32, 128, 128))
    metas, geoms, t0 = [], {}, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (v, meta, geom) in enumerate(ex.map(process, jobs, chunksize=4)):
            meta["row"] = i
            metas.append(meta)
            if v is not None:
                vols[i] = v
                geoms[meta["SeriesInstanceUID"]] = geom
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    vols.flush()
    s = pd.DataFrame(metas)
    s.to_csv(os.path.join(out_dir, "series.csv"), index=False)
    with open(os.path.join(out_dir, "geometry.json"), "w") as f:
        json.dump(geoms, f)
    print(f"-> {int((s.status == 'ok').sum())}/{len(s)} series in {time.time() - t0:.0f}s;",
          s[s.status == "ok"].groupby("seq")[["slab", "n_slices"]].median().to_dict("index"))


if __name__ == "__main__":
    main()
