"""ROI boxes on axial series, by registration from the study's sagittal and coronal boxes.

    python -m crops.axial_boxes --pred round3_pred --out round3_axial

The ROI U-Net never sees axial series; their boxes are the sagittal and coronal
predictions of the same knee (same study, same predicted side) mapped into the
axial grid and intersected in patient space (`roi.consensus.registered_box`).
"""

import argparse
import os

import pandas as pd

from params import OUTPUT_FOLDER
from roi.consensus import add_confidence, registered_box
from roi.labels import BOX_COLS, LABEL_DIR
from roi.laterality import series_side


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="round3_pred")
    p.add_argument("--out", default="round3_axial")
    a = p.parse_args()
    side = series_side()
    pred = pd.read_csv(os.path.join(LABEL_DIR, f"{a.pred}.csv"))
    pred["knee"] = pred.StudyInstanceUID + "|" + pred.SeriesInstanceUID.map(side).fillna("?")
    pred = add_confidence(pred.assign(StudyInstanceUID=pred.knee))
    axial = pd.read_csv(os.path.join(OUTPUT_FOLDER, "roi", "volumes_axial.csv"))
    axial["knee"] = axial.StudyInstanceUID + "|" + axial.SeriesInstanceUID.map(side).fillna("?")
    rows = []
    by_knee = dict(tuple(pred.groupby("knee")))
    for t in axial.itertuples():
        src = by_knee.get(t.knee)
        if src is None:
            continue
        for cls, s in src.groupby("class"):
            box = registered_box(t, s)
            if box is not None:
                rows.append(dict(SeriesInstanceUID=t.SeriesInstanceUID, StudyInstanceUID=t.StudyInstanceUID,
                                 plane="Axial", **{"class": cls}, source="registered",
                                 confidence=float(s.confidence.max()), **box))
    out = pd.DataFrame(rows)
    out[["SeriesInstanceUID", "StudyInstanceUID", "plane", "class", "source", "confidence"] + BOX_COLS].to_csv(
        os.path.join(LABEL_DIR, f"{a.out}.csv"), index=False)
    print(f"-> {len(out)} axial boxes on {out.SeriesInstanceUID.nunique()} / {len(axial)} axial series; "
          f"per class {out['class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
