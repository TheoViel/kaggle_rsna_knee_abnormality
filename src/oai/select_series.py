"""OAI baseline series to download: the 5 knee sequences of every knee with a baseline MOAKS reading, run from src/.

    python -m oai.select_series

Needs input/oai/images/baseline/image03.txt (the NDA package manifest, from
`downloadcmd -dp <package> --file-regex '.*\\.(txt|pdf)$'`) and input/oai/Semi-Quant Scoring_ASCII/
kMRI_SQ_MOAKS_BICL00.txt. Writes input/oai/images/baseline_moaks_series_s3.txt (the S3 links, the
`downloadcmd -t` input) and baseline_moaks_series.csv (participant, side, sequence per link; read by oai.prep).
"""

import os

import numpy as np
import pandas as pd

from params import INPUT_FOLDER

SEQS = ["SAG_IW_TSE", "COR_IW_TSE", "SAG_3D_DESS", "COR_MPR", "AX_MPR"]


def main():
    img = os.path.join(INPUT_FOLDER, "oai", "images")
    d = pd.read_csv(os.path.join(img, "baseline", "image03.txt"), sep="\t", skiprows=[1], low_memory=False).copy()
    m = pd.read_csv(os.path.join(INPUT_FOLDER, "oai", "Semi-Quant Scoring_ASCII", "kMRI_SQ_MOAKS_BICL00.txt"), sep="|", low_memory=False)
    m["side"] = m.SIDE.astype(str).str.extract(r"(Right|Left)", expand=False).str.upper()
    knees = set(zip(m.ID.astype(str), m.side))
    ex = d.image_description.str.extract(r"^(" + "|".join(SEQS) + r")_(RIGHT|LEFT)$")
    d = d.assign(seq=ex[0], side=ex[1])
    inknee = np.array([(str(a), b) in knees for a, b in zip(d.src_subject_id, d.side)])
    sel = d[d.seq.notna().to_numpy() & inknee]
    print("series:", len(sel), "| knees:", sel[["src_subject_id", "side"]].drop_duplicates().shape[0], "of", len(knees))
    sel.image_file.to_csv(os.path.join(img, "baseline_moaks_series_s3.txt"), index=False, header=False)
    sel[["src_subject_id", "side", "seq", "image_description", "image_file", "interview_date"]].to_csv(
        os.path.join(img, "baseline_moaks_series.csv"), index=False)


if __name__ == "__main__":
    main()
