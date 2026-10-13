"""Turn the raw series into the uint8 cache training reads, run from src/.

    python prepare_cache.py --workers 96

The raw arrays are 523 GB of uint16 on NFS: too slow to read every epoch and too
big to sit in page cache next to four training processes. Each series is
normalised once here and stored at CACHE_SIZE x CACHE_SIZE, which brings the set
to ~100 GB, all of which the OS can keep in RAM.

Per series:
  - intensities are clipped to the volume's [0.5, 99.5] percentiles and scaled
    to 0-255. MR intensities have no absolute meaning, so per-series is the only
    normalisation that transfers between scanners;
  - non-square slices are zero-padded to square (centred) so anatomy keeps its
    aspect ratio, then resized with antialiasing;
  - series longer than CACHE_MAX_SLICES (the 3D isotropic acquisitions, ~2%) are
    subsampled uniformly along z. Slice order is kept, it is already sorted
    along the slice normal by the preparation notebook.

`preprocess_volume` is the function inference must reuse on DICOM-decoded volumes.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import CACHE_DIR, CACHE_MAX_SLICES, CACHE_SIZE, SERIES_INDEX_CSV


def preprocess_volume(vol, size=CACHE_SIZE, max_slices=CACHE_MAX_SLICES):
    """(Z, H, W) raw MR -> (Z', size, size) uint8."""
    import torch
    import torch.nn.functional as F

    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None]
    if len(vol) > max_slices:
        keep = np.round(np.linspace(0, len(vol) - 1, max_slices)).astype(int)
        vol = vol[keep]

    sample = vol[:, ::4, ::4]
    lo, hi = np.percentile(sample, [0.5, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)

    z, h, w = vol.shape
    if h != w:
        s = max(h, w)
        pad = np.zeros((z, s, s), dtype=np.float32)
        y0, x0 = (s - h) // 2, (s - w) // 2
        pad[:, y0:y0 + h, x0:x0 + w] = vol
        vol = pad

    t = torch.from_numpy(vol)[:, None]
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                          antialias=True, align_corners=False)
    return (t[:, 0].clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def _convert(job):
    import torch

    torch.set_num_threads(1)
    src, dst = job
    if os.path.exists(dst):
        return "skipped", dst
    try:
        out = preprocess_volume(np.load(src))
        tmp = dst + ".tmp.npy"
        np.save(tmp, out)
        os.replace(tmp, dst)
        return "ok", dst
    except Exception as e:  # noqa: BLE001 -- report and keep going
        return f"failed: {e}", dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    idx = pd.read_csv(SERIES_INDEX_CSV)
    if args.limit:
        idx = idx.sample(args.limit, random_state=0)
    jobs = [("../" + p, CACHE_DIR + f"{s}.npy")
            for p, s in zip(idx.array_path, idx.SeriesInstanceUID)]

    t0, counts = time.time(), {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (status, dst) in enumerate(ex.map(_convert, jobs, chunksize=4)):
            key = status if not status.startswith("failed") else "failed"
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                print(status, dst, flush=True)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(jobs)} {counts} {time.time() - t0:.0f}s", flush=True)
    print(f"done {counts} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
