"""SKM-TEA scans (input/skm_tea/dicom/MTR_*.tar.gz) -> our cache / volumes / metadata, run from src/.

    python -m skm.prep --workers 32

Each archive holds one 3D sagittal qDESS acquisition (two echoes, 160 x 0.8 mm slices, plus a T2 map).
The second echo (T2-weighted, bright fluid, the closest to a fluid-sensitive clinical sagittal) is
run through the same path as `oai.prep`: slice order, ~3 mm slab averaging, 384 px cache, 32 x 128 x 128
volume in the training left-right convention. `slab` and `n_raw` (the 0.8 mm slice count) let the
annotation boxes, given in the raw slice grid, be mapped onto the slabs.

Writes output/skm/: cache_384/<uid>.npy, volumes.npy, series.csv, geometry.json.
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

from oai.prep import slab_average
from params import INPUT_FOLDER, OUTPUT_FOLDER

SKM_DIR = os.path.join(OUTPUT_FOLDER, "skm")


def process(job):
    import pydicom
    import torch

    from infer.prep import slice_order
    from knee_registration import geometry_from_slices
    from prepare_cache import preprocess_volume
    from roi.prepare import cache_to_volume

    torch.set_num_threads(1)
    path, cache_dir = job
    scan = os.path.basename(path).split(".")[0]
    meta = dict(scan_id=scan, StudyInstanceUID=f"SKM_{scan}", plane="Sagittal", Fluid_Sensitive=1)
    try:
        with tarfile.open(path) as t:
            dss = [pydicom.dcmread(io.BytesIO(t.extractfile(m).read())) for m in t.getmembers() if m.isfile()]
        dss = [d for d in dss if "PixelData" in d and "DESS" in str(d.get("SeriesDescription", ""))
               and "T2 map" not in str(d.get("SeriesDescription", "")) and str(d.get("EchoNumbers", "")) == "2"]
        order, _ = slice_order(dss)
        dss = [dss[i] for i in order]
        vol = np.stack([d.pixel_array for d in dss]).astype(np.float32)
        n_raw = len(dss)
        normal = np.cross(np.array(dss[0].ImageOrientationPatient[:3], float), np.array(dss[0].ImageOrientationPatient[3:], float))
        pos = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in dss])
        spacing = float(np.median(np.abs(np.diff(pos))))
        # Direction of increasing (sorted) slice index in patient space: +x = patient left (LPS).
        step = (np.array(dss[-1].ImagePositionPatient, float) - np.array(dss[0].ImagePositionPatient, float)) / max(n_raw - 1, 1)
        vol, dss, k = slab_average(vol, dss, spacing)
        uid = str(dss[0].SeriesInstanceUID)
        geom = geometry_from_slices([d.ImageOrientationPatient for d in dss], [d.ImagePositionPatient for d in dss],
                                    dss[0].PixelSpacing, dss[0].get("SliceThickness"))
        cache = preprocess_volume(vol)
        np.save(os.path.join(cache_dir, f"{uid}.npy"), cache)
        volume = cache_to_volume(cache)
        d = np.array(geom["direction"]).reshape(3, 3)
        conv_flip = bool(d[0, 2] > 0)
        if conv_flip:
            volume = np.ascontiguousarray(np.flip(volume, 0))
        meta.update(SeriesInstanceUID=uid, n_slices=int(vol.shape[0]), n_cache=int(cache.shape[0]), rows=int(vol.shape[1]),
                    cols=int(vol.shape[2]), conv_flip=conv_flip, slab=k, n_raw=n_raw, spacing=spacing,
                    slice_step_x=float(step[0]), status="ok")
        return volume, meta, geom
    except Exception as e:  # noqa: BLE001
        meta.update(SeriesInstanceUID=f"failed_{scan}", status=f"failed: {e!r}"[:200])
        return None, meta, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    paths = sorted(os.path.join(INPUT_FOLDER, "skm_tea", "dicom", f) for f in os.listdir(os.path.join(INPUT_FOLDER, "skm_tea", "dicom")))
    cache_dir = os.path.join(SKM_DIR, "cache_384")
    os.makedirs(cache_dir, exist_ok=True)
    vols = np.lib.format.open_memmap(os.path.join(SKM_DIR, "volumes.npy"), mode="w+", dtype=np.uint8, shape=(len(paths), 32, 128, 128))
    metas, geoms, t0 = [], {}, time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (v, meta, geom) in enumerate(ex.map(process, [(pth, cache_dir) for pth in paths])):
            meta["row"] = i
            metas.append(meta)
            if v is not None:
                vols[i] = v
                geoms[meta["SeriesInstanceUID"]] = geom
    vols.flush()
    s = pd.DataFrame(metas)
    s.to_csv(os.path.join(SKM_DIR, "series.csv"), index=False)
    with open(os.path.join(SKM_DIR, "geometry.json"), "w") as f:
        json.dump(geoms, f)
    print(f"-> {int((s.status == 'ok').sum())}/{len(s)} scans in {time.time() - t0:.0f}s;",
          s[s.status == "ok"][["n_raw", "slab", "n_slices", "spacing"]].median().to_dict(),
          "| slice direction +x (towards patient left):", int((s.slice_step_x > 0).sum()), "of", len(s))


if __name__ == "__main__":
    main()
