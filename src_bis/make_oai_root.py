"""A competition-style root of OAI knees, to run the team's b1_c1 (and our models) on image-graded effusion.

    python src_bis/make_oai_root.py OUT N          (N knees per MOAKS effusion grade: >= 2, mild, 0; seed 0)

OUT/input/rsna-knee-abnormality-detection/{test.csv, test_series.csv, test_series/<knee>/<series>/<n>.dcm}, from the
baseline per-series tar.gz archives (members renamed to .dcm). Series columns as the competition's:
  SAG_IW_TSE  Sagittal, fluid-sensitive, fat-suppressed (first sagittal: B1's fluid slot)
  SAG_3D_DESS Sagittal, fluid-sensitive (water excitation)
  COR_IW_TSE  Coronal, not fluid-sensitive / fat-suppressed
  COR_MPR, AX_MPR (DESS reformats) Coronal / Axial, fluid-sensitive
Knee ids are OAI_<id>_<SIDE>, as in output/oai/labels_moaks00_v4.csv.
"""

import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE = "/data/rsna_knee"
SEQ = {"SAG_IW_TSE": ("Sagittal", 1, 1), "SAG_3D_DESS": ("Sagittal", 1, 1), "COR_IW_TSE": ("Coronal", 0, 0),
       "COR_MPR": ("Coronal", 1, 1), "AX_MPR": ("Axial", 1, 1)}
out, n = sys.argv[1], int(sys.argv[2])
root = os.path.join(out, "input", "rsna-knee-abnormality-detection")
lab = pd.read_csv(f"{BASE}/output/oai/labels_moaks00_v4.csv").set_index("StudyInstanceUID").Effusion
rng = np.random.default_rng(0)
knees = []
for g in (1.0, 0.25, 0.0):
    ids = lab[lab == g].index.to_numpy()
    knees += list(ids[rng.permutation(len(ids))[:n]])
idx = pd.read_csv(f"{BASE}/input/oai/images/baseline_moaks_series.csv")
idx["knee"] = "OAI_" + idx.src_subject_id.astype(str) + "_" + idx.side
idx = idx[idx.knee.isin(set(knees)) & idx.seq.isin(SEQ)].copy()
idx["order"] = idx.seq.map(list(SEQ).index)
idx = idx.sort_values(["knee", "order"])
idx["series"] = idx.image_file.str.split("/").str[-1].str.replace(".tar.gz", "", regex=False)


def extract(r):
    d = os.path.join(root, "test_series", r.knee, r.series)
    if os.path.isdir(d) and os.listdir(d):
        return
    os.makedirs(d, exist_ok=True)
    src = os.path.join(BASE, "input/oai/images/baseline/image03", r.image_file.split("submission_60350/")[1])
    with tarfile.open(src) as t:
        for m in t.getmembers():
            if m.isfile():
                with open(os.path.join(d, os.path.basename(m.name) + ".dcm"), "wb") as f:
                    f.write(t.extractfile(m).read())


with ThreadPoolExecutor(16) as ex:
    list(ex.map(extract, idx.itertuples()))
rows = [dict(StudyInstanceUID=r.knee, SeriesInstanceUID=r.series, Fluid_Sensitive=SEQ[r.seq][1],
             Fat_Suppression=SEQ[r.seq][2], Anatomical_Plane=SEQ[r.seq][0]) for r in idx.itertuples()]
pd.DataFrame(rows).to_csv(os.path.join(root, "test_series.csv"), index=False)
pd.DataFrame({"StudyInstanceUID": sorted(idx.knee.unique())}).to_csv(os.path.join(root, "test.csv"), index=False)
print(root, idx.knee.nunique(), "knees", len(rows), "series")
