"""fastMRI clinical knee DICOMs (input/fastmri/dicom/knee_mri_clinical_seq_batch*/<patient>/<study>/<series>/*.dcm)
-> a flat series tree + series list for the inference / prep modules; run from src/.

    python -m fastmri.dicom_index [--batches 2]

The DICOM folder names (MR4_53c76c27, ...) are not guaranteed unique across studies, and the pipeline names caches
after them, so a symlink tree with unique ids is built: input/fastmri/dicom_flat/FMD_<study>/FMD_<study>_<series>.
One header per series gives the plane (majority orientation), the description and the slice count; localisers and
scouts (< 10 slices, or "loc" / "scout" / "survey" in the description) are left out. Writes
input/fastmri/dicom_series.csv (StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, n_files, description, batch).
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from params import INPUT_FOLDER

ROOT = os.path.join(INPUT_FOLDER, "fastmri", "dicom")
FLAT = os.path.join(INPUT_FOLDER, "fastmri", "dicom_flat")
SKIP = ("loc", "scout", "survey", "calib", "3pl")


def header(job):
    import pydicom

    folder, study_id, series_id, batch = job
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    row = dict(StudyInstanceUID=study_id, SeriesInstanceUID=series_id, n_files=len(files), batch=batch, folder=folder,
               Anatomical_Plane="", description="")
    if not files:
        return row
    try:
        d = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
        row["description"] = str(d.get("SeriesDescription", ""))
        iop = np.array(d.ImageOrientationPatient, float)
        n = np.abs(np.cross(iop[:3], iop[3:]))
        row["Anatomical_Plane"] = ["Sagittal", "Coronal", "Axial"][int(np.argmax(n))]
    except Exception:  # noqa: BLE001
        pass
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", default="1,2")
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    jobs = []
    for b in a.batches.split(","):
        base = os.path.join(ROOT, "knee_mri_clinical_seq" + ("" if b == "1" else f"_batch{b}"))     # batch 1: no suffix
        for study in sorted(glob.glob(os.path.join(base, "*", "*"))):
            sid = "FMD_" + os.path.basename(study).replace("study_", "")
            for ser in sorted(glob.glob(os.path.join(study, "*"))):
                if os.path.isdir(ser):
                    jobs.append((ser, sid, f"{sid}_{os.path.basename(ser)}", b))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(header, jobs, chunksize=16))
    df = pd.DataFrame(rows)
    desc = df.description.str.lower()
    keep = (df.n_files >= 10) & (df.Anatomical_Plane != "") & ~desc.apply(lambda s: any(k in s for k in SKIP))
    dup = df.duplicated("SeriesInstanceUID", keep=False)
    print(f"{len(df)} series in {df.StudyInstanceUID.nunique()} studies; kept {int((keep & ~dup).sum())} "
          f"({int(dup.sum())} duplicate ids dropped)")
    df = df[keep & ~dup]
    for r in df.itertuples():
        d = os.path.join(FLAT, r.StudyInstanceUID)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, r.SeriesInstanceUID)
        if not os.path.islink(link):
            os.symlink(os.path.abspath(r.folder), link)
    old = os.path.join(INPUT_FOLDER, "fastmri", "dicom_series.csv")
    if os.path.exists(old) and set(a.batches.split(",")) != {"1", "2"}:
        prev = pd.read_csv(old, dtype={"batch": str})
        df = pd.concat([prev[~prev.batch.isin(a.batches.split(","))], df], ignore_index=True)
    df.drop(columns=["folder"]).to_csv(old, index=False)
    print(df.groupby("Anatomical_Plane").size().to_dict(), "| studies:", df.StudyInstanceUID.nunique(),
          "| series per study:", df.groupby("StudyInstanceUID").size().median())
    print("top descriptions:", df.description.str.lower().value_counts().head(12).to_dict())


if __name__ == "__main__":
    main()
