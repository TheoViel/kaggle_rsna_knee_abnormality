"""MRNet (input/mrnet, Stanford knee MRI, .npy volumes) -> our cache / volumes / metadata, run from src/.

    python -m mrnet.prep --workers 32

MRNet ships each exam as three preprocessed uint8 arrays (axial / coronal / sagittal, slices x 256 x 256),
without DICOM geometry. They go through the same operations as our series (`prepare_cache.preprocess_volume`
-> 384 px cache, `roi.prepare.cache_to_volume` -> 32 x 128 x 128). The left-right convention cannot be
checked without geometry, so no `conv_flip`: the side classifier then canonicalises each series to a right
knee whatever its stored orientation (a mirrored right knee reads as a left knee and is flipped back), which
is all the crops need -- MRNet labels have no side.

Writes output/mrnet/: cache_384/<uid>.npy, volumes.npy, series.csv, labels.csv (StudyInstanceUID MRNET_<case>,
mrnet_acl, mrnet_meniscus, mrnet_abnormal, split).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import INPUT_FOLDER, OUTPUT_FOLDER

MRNET_IN = os.path.join(INPUT_FOLDER, "mrnet")
MRNET_DIR = os.path.join(OUTPUT_FOLDER, "mrnet")
PLANES = {"sagittal": "Sagittal", "coronal": "Coronal", "axial": "Axial"}


def process(job):
    import torch

    from prepare_cache import preprocess_volume
    from roi.prepare import cache_to_volume

    torch.set_num_threads(1)
    path, split, plane, case, cache_dir = job
    uid = f"MRNET_{case}_{plane}"
    meta = dict(SeriesInstanceUID=uid, StudyInstanceUID=f"MRNET_{case}", case=case, split=split, plane=PLANES[plane],
                Fluid_Sensitive=1 if plane != "coronal" else 0)        # MRNet: sag T2 FS, axial PD FS, coronal T1
    try:
        vol = np.load(path).astype(np.float32)
        cache = preprocess_volume(vol)
        np.save(os.path.join(cache_dir, f"{uid}.npy"), cache)
        meta.update(n_slices=int(vol.shape[0]), n_cache=int(cache.shape[0]), rows=int(vol.shape[1]), cols=int(vol.shape[2]),
                    conv_flip=False, status="ok")
        return cache_to_volume(cache), meta
    except Exception as e:  # noqa: BLE001
        meta.update(status=f"failed: {e!r}"[:200])
        return None, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    cache_dir = os.path.join(MRNET_DIR, "cache_384")
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(os.path.join(MRNET_IN, split, plane, f), split, plane, f[:-4], cache_dir)
            for split in ("train", "valid") for plane in PLANES for f in sorted(os.listdir(os.path.join(MRNET_IN, split, plane)))]
    vols = np.lib.format.open_memmap(os.path.join(MRNET_DIR, "volumes.npy"), mode="w+", dtype=np.uint8, shape=(len(jobs), 32, 128, 128))
    metas, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (v, meta) in enumerate(ex.map(process, jobs, chunksize=8)):
            meta["row"] = i
            metas.append(meta)
            if v is not None:
                vols[i] = v
    vols.flush()
    s = pd.DataFrame(metas)
    s.to_csv(os.path.join(MRNET_DIR, "series.csv"), index=False)
    lab = None
    for split in ("train", "valid"):
        parts = [pd.read_csv(os.path.join(MRNET_IN, f"{split}-{t}.csv"), dtype={"case": str}).rename(columns={"label": f"mrnet_{t}"})
                 .set_index("case") for t in ("acl", "meniscus", "abnormal")]
        d = pd.concat(parts, axis=1).assign(split=split)
        lab = d if lab is None else pd.concat([lab, d])
    lab.index = "MRNET_" + lab.index
    lab.rename_axis("StudyInstanceUID").to_csv(os.path.join(MRNET_DIR, "labels.csv"))
    print(f"-> {int((s.status == 'ok').sum())}/{len(s)} series in {time.time() - t0:.0f}s; {len(lab)} exams;",
          s[s.status == "ok"].groupby("plane").n_slices.median().to_dict())


if __name__ == "__main__":
    main()
