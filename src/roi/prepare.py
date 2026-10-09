"""Downscaled volumes for the ROI U-Net, run from src/.

    python -m roi.prepare                                        # sagittal + coronal -> volumes
    python -m roi.prepare --planes Axial --name volumes_axial    # axial, for the side classifier

Every sagittal and coronal series becomes a (D, S, S) = (32, 128, 128) uint8
volume, resampled from the uint8 cache (`prepare_cache.py`: per-series
normalised, zero-padded to a centred square, 384 px, at most 64 slices).
In-plane the grid is the padded square, like the attention maps; along z the
slices are resampled to D whatever their number, so a voxel is not isotropic and
its size differs between series -- the network sees index space, and boxes are
converted to and from it exactly (`roi.geometry`).

Output: output/roi/volumes.npy (N, D, S, S) uint8, memory-mappable, and output/roi/volumes.csv
(row -> series, with the shapes the coordinate conversions need).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import CACHE_DIR, OUTPUT_FOLDER, SERIES_INDEX_CSV

ROI_DIR = os.path.join(OUTPUT_FOLDER, "roi")
DEPTH, SIZE = 32, 128
ROI_PLANES = ("Sagittal", "Coronal")


def cache_to_volume(cache):
    """(n, 384, 384) uint8 cache array -> (DEPTH, SIZE, SIZE) uint8 volume.

    Also what inference applies to freshly decoded series, so test volumes go
    through exactly the training operations.
    """
    import torch
    import torch.nn.functional as F

    v = torch.from_numpy(np.asarray(cache, dtype=np.float32))[None, None]   # 1 1 n 384 384
    v = F.interpolate(v, size=(v.shape[2], SIZE, SIZE), mode="area")
    v = F.interpolate(v, size=(DEPTH, SIZE, SIZE), mode="trilinear", align_corners=False)
    return v[0, 0].round().clamp(0, 255).to(torch.uint8).numpy()


def _resample(path):
    import torch

    torch.set_num_threads(1)
    cache = np.load(path)
    return cache_to_volume(cache), int(cache.shape[0])


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--planes", default=",".join(ROI_PLANES))
    p.add_argument("--name", default="volumes", help="output/roi/<name>.npy / .csv")
    a = p.parse_args()
    os.makedirs(ROI_DIR, exist_ok=True)
    idx = pd.read_csv(SERIES_INDEX_CSV)
    idx = idx[idx.Anatomical_Plane.isin(a.planes.split(","))].reset_index(drop=True)
    paths = [os.path.join(CACHE_DIR, f"{s}.npy") for s in idx.SeriesInstanceUID]
    vol = np.lib.format.open_memmap(os.path.join(ROI_DIR, f"{a.name}.npy"), mode="w+",
                                    dtype=np.uint8, shape=(len(idx), DEPTH, SIZE, SIZE))
    n_cache = np.zeros(len(idx), int)
    with ProcessPoolExecutor(96) as ex:
        for i, (v, n) in enumerate(ex.map(_resample, paths, chunksize=16)):
            vol[i] = v
            n_cache[i] = n
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(idx)}", flush=True)
    vol.flush()
    idx["row"] = np.arange(len(idx))
    idx["n_cache"] = n_cache
    idx[["row", "SeriesInstanceUID", "StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive",
         "n_slices", "n_cache", "rows", "cols"]].to_csv(os.path.join(ROI_DIR, f"{a.name}.csv"), index=False)
    print(f"-> {len(idx)} volumes {DEPTH}x{SIZE}x{SIZE} in {ROI_DIR}")


if __name__ == "__main__":
    main()
