"""MRNet exam labels (output/mrnet/labels.csv) -> our findings, run from src/.

    python -m mrnet.labels                                    # labels_graded.csv (+ the two combined files below)
    python -m mrnet.labels --men_side <preds.csv> --name labels_menpl
    python -m mrnet.labels --men_soft 0.75,0.4 --name labels_mensoft
    python -m mrnet.labels --pl --name labels_pl              # needs crops.predict --crops mrnet_s1n8 for PL_RUNS
    python -m mrnet.labels --abn --name labels_abn            # + "<ROI>|abnormal" auxiliary targets (group all_abn)

MRNet has three binary report-derived labels per exam: abnormal, ACL tear, meniscal tear (no side,
no grade). What they say about our findings:
- ACL: the ACL label (positives are any tear, possibly partial: no grade to map from);
- menisci: no meniscal tear -> both 0; a tear -> side unknown, NaN. Variants for the positives:
  `--men_side <csv>` splits them with a model's medial / lateral predictions (the likelier side 1, the
  other its own probability); `--men_soft m,l` gives fixed soft targets (medial m, lateral l: about the
  medial / lateral tear prior, 73% of positives look medial to our menisci model);
- abnormal = 0 (a normal knee) -> every finding 0. Otherwise MCL, OA, Baker's are unknown (NaN), except
  with `--pl`: pseudo-labels from our reference models' predictions on the MRNet crops (mean of the
  recipe's models per finding, PL_RUNS), with the MRNet labels on top (ACL label; menisci as for
  `--men_side` with the same predictions; normal knees all 0).
Writes output/mrnet/<name>.csv. The default (graded) run also writes the combined label files for
training with several external sets at once: output/ext/skm_mrnet_labels_graded.csv (SKM-TEA + MRNet)
and output/ext/oai4_mrnet_labels.csv (OAI graded v4 + MRNet).
"""

import argparse
import os

import numpy as np
import pandas as pd

from params import OUTPUT_FOLDER

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Baker's",
            "Effusion", "Synovitis", "Contusion", "Fracture"]
MEN = ["Medial Meniscus", "Lateral Meniscus"]
# Pseudo-label sources (non-fullfit reference runs, dataset_2 setups): finding -> runs averaged.
PL_RUNS = {"ACL": [79, 75], "MCL": [80, 75], "Medial Meniscus": [81, 75], "Lateral Meniscus": [81, 75],
           "Medial OA": [82, 75], "Lateral OA": [82], "PF OA": [82], "Baker's": [84, 75]}
PL_DIR = "../logs/crops_run/2026-10-09/"


def side_split(out, pos, pr):
    """Meniscus-positive exams: the likelier side (per `pr`) 1, the other side its predicted probability."""
    top = pr.loc[pos, MEN].idxmax(axis=1)
    for c in MEN:
        out.loc[pos, c] = np.where(top == c, 1.0, pr.loc[pos, c])


def pseudo_labels(lab):
    preds = {r: pd.read_csv(os.path.join(PL_DIR, str(r), "mrnet_s1n8_study_preds.csv")).set_index("StudyInstanceUID")
             for r in sorted({r for rs in PL_RUNS.values() for r in rs})}
    pl = pd.DataFrame({c: sum(preds[r][c] for r in rs) / len(rs) for c, rs in PL_RUNS.items()})
    raw = lab.reindex(pl.index)
    pl["ACL"] = raw.mrnet_acl.astype(float)
    pl.loc[raw.mrnet_meniscus == 0, MEN] = 0.0
    side_split(pl, raw.mrnet_meniscus != 0, pl.copy())
    pl.loc[raw.mrnet_abnormal == 0] = 0.0
    return pl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--men_side", default="", help="csv of StudyInstanceUID x meniscus predictions for positive exams")
    p.add_argument("--men_soft", default="", help="'m,l': fixed medial / lateral targets for positive exams")
    p.add_argument("--pl", action="store_true", help="pseudo-labels for the unlabelled findings (PL_RUNS)")
    p.add_argument("--abn", action="store_true", help="add '<ROI>|abnormal' columns (the exam's abnormal label)")
    p.add_argument("--name", default="labels_graded")
    a = p.parse_args()
    d = os.path.join(OUTPUT_FOLDER, "mrnet")
    lab = pd.read_csv(os.path.join(d, "labels.csv")).set_index("StudyInstanceUID")
    if a.pl:
        out = pseudo_labels(lab)
    else:
        out = pd.DataFrame(np.nan, index=lab.index, columns=FINDINGS)
        out.loc[lab.mrnet_abnormal == 0] = 0.0
        out["ACL"] = lab.mrnet_acl.astype(float)
        out.loc[lab.mrnet_meniscus == 0, MEN] = 0.0
        pos = lab.mrnet_meniscus == 1
        if a.men_side:
            side_split(out, pos, pd.read_csv(a.men_side).set_index("StudyInstanceUID").reindex(lab.index))
        elif a.men_soft:
            out.loc[pos, MEN] = [float(x) for x in a.men_soft.split(",")]
    if a.abn:
        for c in FINDINGS[:8]:
            out[f"{c}|abnormal"] = lab.mrnet_abnormal.reindex(out.index).astype(float)
    out.rename_axis("StudyInstanceUID").to_csv(os.path.join(d, f"{a.name}.csv"))
    print(out.describe().T[["count", "mean"]].round(3).to_string())
    if a.name == "labels_graded":
        os.makedirs(os.path.join(OUTPUT_FOLDER, "ext"), exist_ok=True)
        skm = pd.read_csv(os.path.join(OUTPUT_FOLDER, "skm", "labels_graded.csv")).set_index("StudyInstanceUID")[FINDINGS]
        pd.concat([skm, out]).rename_axis("StudyInstanceUID").to_csv(os.path.join(OUTPUT_FOLDER, "ext", "skm_mrnet_labels_graded.csv"))
        oai = pd.read_csv(os.path.join(OUTPUT_FOLDER, "oai", "labels_moaks00_v4.csv")).set_index("StudyInstanceUID")[FINDINGS]
        pd.concat([oai, out]).rename_axis("StudyInstanceUID").to_csv(os.path.join(OUTPUT_FOLDER, "ext", "oai4_mrnet_labels.csv"))


if __name__ == "__main__":
    main()
