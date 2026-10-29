"""Training DICOMs -> raw NumPy arrays + metadata, run from src/.

    python prepare_data.py --workers 64

For every series of input/train_series.csv (input/train_series/<study>/<series>/*.dcm):
  - pixels -> output/series_npy/<SeriesInstanceUID>.npy, (n_slices, rows, cols) in the native dtype (uint16 /
    int16), no rescaling, windowing or resampling. Slices are ordered along the slice normal (`dicom_io.slice_order`,
    the order inference uses). A series whose slices differ in shape or dtype is reported as failed;
  - every DICOM tag but PixelData, one row per slice in stacked order (slice_idx) -> output/metadata_shards/
    <SeriesInstanceUID>.pkl (the geometry `knee_registration.series_geometry` reads), concatenated into
    output/metadata.csv;
  - output/series_index.csv, one row per series: array path, shape, dtype and the csv's plane / fluid / fat flags.
Resumable: series that already have both outputs are skipped. `prepare_cache.py` then builds the uint8 training cache.
"""

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dicom_io import slice_order
from params import INPUT_FOLDER, OUTPUT_FOLDER, SERIES_INDEX_CSV

ARR_DIR = os.path.join(OUTPUT_FOLDER, "series_npy")
SHARD_DIR = os.path.join(OUTPUT_FOLDER, "metadata_shards")
FLAGS = ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]


def _clean(v):
    """A pydicom value as something CSV friendly, without losing information."""
    import pydicom

    if isinstance(v, pydicom.multival.MultiValue):
        return json.dumps([_clean(x) for x in v])
    if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
        return float(v)
    if isinstance(v, pydicom.valuerep.IS):
        return int(v)
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def slice_meta(ds):
    row = {}
    for e in ds:
        if e.tag == 0x7FE00010:   # PixelData
            continue
        key = e.keyword or f"Tag_{e.tag.group:04X}{e.tag.element:04X}"
        row[key] = f"<sequence of {len(e.value)}>" if e.VR == "SQ" else _clean(e.value)
    row["TransferSyntax"] = ds.file_meta.TransferSyntaxUID.name
    return row


def convert_series(job):
    import pydicom

    study_uid, series_uid = job
    arr_path = os.path.join(ARR_DIR, f"{series_uid}.npy")
    shard_path = os.path.join(SHARD_DIR, f"{series_uid}.pkl")
    if os.path.exists(arr_path) and os.path.exists(shard_path):
        return "skipped", series_uid, None
    try:
        folder = os.path.join(INPUT_FOLDER, "train_series", study_uid, series_uid)
        dss = [pydicom.dcmread(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".dcm")]
        order, _ = slice_order(dss)
        dss = [dss[i] for i in order]
        arrs = [d.pixel_array for d in dss]
        arrs = [a[0] if a.ndim == 3 and a.shape[0] == 1 else a for a in arrs]   # single-frame multiframe
        if len({(a.shape, a.dtype) for a in arrs}) != 1:
            raise ValueError(f"inconsistent slices: {sorted({(a.shape, str(a.dtype)) for a in arrs})}")
        np.save(arr_path[:-4] + ".tmp.npy", np.stack(arrs))
        os.replace(arr_path[:-4] + ".tmp.npy", arr_path)
        rel = os.path.relpath(arr_path, os.path.join(OUTPUT_FOLDER, ".."))
        pd.DataFrame([dict(slice_meta(d), slice_idx=i, n_slices=len(dss), array_path=rel)
                      for i, d in enumerate(dss)]).to_pickle(shard_path)
        return "ok", series_uid, None
    except Exception as ex:  # noqa: BLE001
        return "failed", series_uid, repr(ex)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    for d in (ARR_DIR, SHARD_DIR):
        os.makedirs(d, exist_ok=True)
    series = pd.read_csv(os.path.join(INPUT_FOLDER, "train_series.csv"))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(convert_series, zip(series.StudyInstanceUID, series.SeriesInstanceUID), chunksize=4))
    print(f"{pd.Series([r[0] for r in results]).value_counts().to_dict()} in {time.time() - t0:.0f}s")
    for _, uid, err in [r for r in results if r[0] == "failed"][:20]:
        print("FAILED", uid, err)

    shards = [os.path.join(SHARD_DIR, f"{s}.pkl") for s in series.SeriesInstanceUID]
    meta = pd.concat([pd.read_pickle(p) for p in shards if os.path.exists(p)], ignore_index=True)
    front = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "slice_idx", "n_slices", "array_path"]
    meta = meta[front + [c for c in meta.columns if c not in front]]
    meta.merge(series[["SeriesInstanceUID"] + FLAGS], on="SeriesInstanceUID", how="left").to_csv(
        os.path.join(OUTPUT_FOLDER, "metadata.csv"), index=False)
    idx = (meta.groupby("SeriesInstanceUID", sort=False)
           .agg(StudyInstanceUID=("StudyInstanceUID", "first"), array_path=("array_path", "first"),
                n_slices=("n_slices", "first"), rows=("Rows", "first"), cols=("Columns", "first"),
                bits_stored=("BitsStored", "first"), pixel_representation=("PixelRepresentation", "first"))
           .reset_index())
    idx["pixel_dtype"] = np.where(idx.pixel_representation == 1, "int16", "uint16")
    idx.merge(series[["SeriesInstanceUID"] + FLAGS], on="SeriesInstanceUID").to_csv(SERIES_INDEX_CSV, index=False)
    print(f"-> {len(meta)} slices, {len(idx)} series: {SERIES_INDEX_CSV}")


if __name__ == "__main__":
    main()
