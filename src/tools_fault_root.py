"""Fault-injection test root for the inference notebooks, run from src/.

    python tools_fault_root.py OUT [--per_mode 2 --clean 6 --seed 0]

Copies training studies into OUT/input/rsna-knee-abnormality-detection/{test_series/, test_series.csv, test.csv}
(the layout local/run_nb.py expects) and alters each study in one way the hidden test set might differ from ours:
  rle / j2k / jpeg8   every series re-encoded (RLE lossless, JPEG 2000 lossless, JPEG baseline 8-bit lossy)
  nogeom_all          ImageOrientationPatient / ImagePositionPatient removed from every slice of every series
  nogeom_one          ... from one sagittal series only
  nospacing           PixelSpacing removed everywhere
  axial_only          sagittal and coronal series removed (folders and csv rows)
  single_series       one sagittal series kept, the rest removed
  multiframe          every series written as one multi-frame file (per-frame positions in functional groups)
  rgb                 every series as RGB (SamplesPerPixel 3)
  blank_plane         Anatomical_Plane empty in the csv for every series
  dup_positions       every slice duplicated at the same position (second echo, InstanceNumber + 1000)
  mono1               MONOCHROME1 (inverted values)
  no_fluid_flags      Fluid_Sensitive / Fat_Suppression columns empty
Clean studies are copied unchanged. Writes OUT/faults.csv (StudyInstanceUID, mode).
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import JPEG2000Lossless, JPEGBaseline8Bit, RLELossless, generate_uid

DATA = "../input"
MODES = ["rle", "j2k", "jpeg8", "nogeom_all", "nogeom_one", "nospacing", "axial_only", "single_series", "multiframe",
         "rgb", "blank_plane", "dup_positions", "mono1", "no_fluid_flags"]


def files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.dcm")))


def to8bit(ds):
    a = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = a.tobytes()
    return a


def encode_jpeg8(ds):
    import io

    from PIL import Image
    from pydicom.encaps import encapsulate
    a = to8bit(ds)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    ds.PixelData = encapsulate([buf.getvalue()])
    ds["PixelData"].VR = "OB"
    ds["PixelData"].is_undefined_length = True
    ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.LossyImageCompression = "01"


def alter_series(src, dst, mode, first_sag):
    os.makedirs(dst, exist_ok=True)
    fs = files(src)
    if mode == "multiframe":
        dss = [pydicom.dcmread(f) for f in fs]
        dss.sort(key=lambda d: int(d.get("InstanceNumber", 0) or 0))
        base = dss[0]
        arr = np.stack([d.pixel_array for d in dss])
        base.NumberOfFrames = len(dss)
        base.PixelData = arr.tobytes()
        if "_pixel_array" in base.__dict__:
            base._pixel_array = None
        shared = Dataset()
        po = Dataset(); po.ImageOrientationPatient = base.ImageOrientationPatient
        pm = Dataset(); pm.PixelSpacing = base.PixelSpacing; pm.SliceThickness = base.get("SliceThickness", 1)
        shared.PlaneOrientationSequence = Sequence([po]); shared.PixelMeasuresSequence = Sequence([pm])
        base.SharedFunctionalGroupsSequence = Sequence([shared])
        per = []
        for d in dss:
            fg = Dataset(); pp = Dataset(); pp.ImagePositionPatient = d.ImagePositionPatient
            fg.PlanePositionSequence = Sequence([pp]); per.append(fg)
        base.PerFrameFunctionalGroupsSequence = Sequence(per)
        for t in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
            if t in base:
                del base[t]
        base.SOPInstanceUID = generate_uid()
        base.save_as(os.path.join(dst, "00001.dcm"), enforce_file_format=True)
        return
    for i, f in enumerate(fs):
        ds = pydicom.dcmread(f)
        if mode == "rle":
            ds.compress(RLELossless)
        elif mode == "j2k":
            ds.compress(JPEG2000Lossless)
        elif mode == "jpeg8":
            encode_jpeg8(ds)
        elif mode == "nogeom_all" or (mode == "nogeom_one" and first_sag):
            for t in ("ImageOrientationPatient", "ImagePositionPatient"):
                if t in ds:
                    del ds[t]
        elif mode == "nospacing":
            if "PixelSpacing" in ds:
                del ds["PixelSpacing"]
        elif mode == "rgb":
            a = to8bit(ds)
            ds.PixelData = np.repeat(a[..., None], 3, axis=-1).tobytes()
            ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
        elif mode == "mono1":
            a = ds.pixel_array
            ds.PixelData = (a.max() - a).astype(a.dtype).tobytes()
            ds.PhotometricInterpretation = "MONOCHROME1"
        ds.save_as(os.path.join(dst, os.path.basename(f)), enforce_file_format=True)
        if mode == "dup_positions":
            ds2 = pydicom.dcmread(f)
            a = ds2.pixel_array
            ds2.PixelData = (a * 0.8).astype(a.dtype).tobytes()
            ds2.InstanceNumber = int(ds2.get("InstanceNumber", i) or i) + 1000
            ds2.SOPInstanceUID = generate_uid()
            ds2.save_as(os.path.join(dst, "echo2_" + os.path.basename(f)), enforce_file_format=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per_mode", type=int, default=2)
    ap.add_argument("--clean", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = os.path.join(a.out, "input", "rsna-knee-abnormality-detection")
    os.makedirs(os.path.join(root, "test_series"), exist_ok=True)
    series = pd.read_csv(os.path.join(DATA, "train_series.csv"), dtype=str)
    studies = series.StudyInstanceUID.drop_duplicates().sample(len(MODES) * a.per_mode + a.clean, random_state=a.seed)
    plan = [(u, m) for u, m in zip(studies, [m for m in MODES for _ in range(a.per_mode)] + ["clean"] * a.clean)]
    rows = []
    for u, mode in plan:
        s = series[series.StudyInstanceUID == u].copy()
        if mode == "axial_only":
            s = s[s.Anatomical_Plane == "Axial"]
        elif mode == "single_series":
            s = s[s.Anatomical_Plane == "Sagittal"].head(1)
        first_sag = s[s.Anatomical_Plane == "Sagittal"].SeriesInstanceUID.head(1).tolist()
        for r in s.itertuples():
            src = os.path.join(DATA, "train_series", u, r.SeriesInstanceUID)
            dst = os.path.join(root, "test_series", u, r.SeriesInstanceUID)
            if mode in ("clean", "axial_only", "single_series", "blank_plane", "no_fluid_flags"):
                shutil.copytree(src, dst)
            else:
                alter_series(src, dst, mode, r.SeriesInstanceUID in first_sag)
        if mode == "blank_plane":
            s["Anatomical_Plane"] = ""
        if mode == "no_fluid_flags":
            s["Fluid_Sensitive"], s["Fat_Suppression"] = "", ""
        rows.append(s)
        print(u[-12:], mode, len(s), "series", flush=True)
    pd.concat(rows).to_csv(os.path.join(root, "test_series.csv"), index=False)
    pd.DataFrame({"StudyInstanceUID": [u for u, _ in plan]}).to_csv(os.path.join(root, "test.csv"), index=False)
    pd.DataFrame(plan, columns=["StudyInstanceUID", "mode"]).to_csv(os.path.join(a.out, "faults.csv"), index=False)


if __name__ == "__main__":
    main()
