"""fastMRI+ annotated knee volumes (input/fastmri/singlecoil) -> our cache / volumes / metadata, run from src/.

    python -m fastmri.prep --workers 32

Each annotated volume (knee_file_list.csv) is one coronal proton-density series (CORPD_FBK, or CORPDFS_FBK with fat
suppression), ~33 slices of 320 x 320 (`reconstruction_esc`, ~0.44 mm per pixel), with no DICOM geometry.
The stored images are upside down (femur at the bottom); the fastMRI+ boxes (knee.csv, x / y / width / height in
pixels) are in the vertically flipped frame. The volumes are flipped here, so images and boxes then share one frame,
and go through the same operations as our series (`prepare_cache.preprocess_volume` -> 384 px cache,
`roi.prepare.cache_to_volume` -> 32 x 128 x 128). No `conv_flip` (no geometry): the side classifier
canonicalises each series, as for MRNet.

Writes output/fastmri/: cache_384/<uid>.npy, volumes.npy, series.csv (SeriesInstanceUID FASTMRI_<file>_cor,
StudyInstanceUID FASTMRI_<file>, file, split, acquisition, Fluid_Sensitive, n_slices, rows, cols, ...).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import INPUT_FOLDER, OUTPUT_FOLDER

FM_IN = os.path.join(INPUT_FOLDER, "fastmri")
FM_DIR = os.path.join(OUTPUT_FOLDER, "fastmri")


def process(job):
    import h5py
    import torch

    from prepare_cache import preprocess_volume
    from roi.prepare import cache_to_volume

    torch.set_num_threads(1)
    path, split, name, cache_dir = job
    uid = f"FASTMRI_{name}_cor"
    meta = dict(SeriesInstanceUID=uid, StudyInstanceUID=f"FASTMRI_{name}", file=name, split=split, plane="Coronal")
    try:
        with h5py.File(path, "r") as f:
            acq = str(f.attrs.get("acquisition", ""))
            vol = f["reconstruction_esc"][:].astype(np.float32)[:, ::-1].copy()        # upright: flip vertically
        vol = vol / max(float(np.percentile(vol, 99.9)), 1e-12) * 1000.0               # scale-free (raw ~1e-4)
        cache = preprocess_volume(vol)
        np.save(os.path.join(cache_dir, f"{uid}.npy"), cache)
        meta.update(acquisition=acq, Fluid_Sensitive=int("FS" in acq), n_slices=int(vol.shape[0]),
                    n_cache=int(cache.shape[0]), rows=int(vol.shape[1]), cols=int(vol.shape[2]), conv_flip=False,
                    status="ok")
        return cache_to_volume(cache), meta
    except Exception as e:  # noqa: BLE001
        meta.update(status=f"failed: {e!r}"[:200])
        return None, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    cache_dir = os.path.join(FM_DIR, "cache_384")
    os.makedirs(cache_dir, exist_ok=True)
    names = pd.read_csv(os.path.join(FM_IN, "knee_file_list.csv"), header=None)[0].tolist()
    jobs = []
    for n in names:
        for split in ("train", "val"):
            path = os.path.join(FM_IN, "singlecoil", f"singlecoil_{split}", f"{n}.h5")
            if os.path.exists(path):
                jobs.append((path, split, n, cache_dir))
                break
    vols = np.lib.format.open_memmap(os.path.join(FM_DIR, "volumes.npy"), mode="w+", dtype=np.uint8, shape=(len(jobs), 32, 128, 128))
    metas, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (v, meta) in enumerate(ex.map(process, jobs, chunksize=4)):
            meta["row"] = i
            metas.append(meta)
            if v is not None:
                vols[i] = v
    vols.flush()
    s = pd.DataFrame(metas)
    s.to_csv(os.path.join(FM_DIR, "series.csv"), index=False)
    print(f"-> {int((s.status == 'ok').sum())}/{len(s)} volumes ({len(names) - len(jobs)} listed but not found) in "
          f"{time.time() - t0:.0f}s;", s[s.status == "ok"].acquisition.value_counts().to_dict(),
          "| slices", s[s.status == "ok"].n_slices.median())


if __name__ == "__main__":
    main()
