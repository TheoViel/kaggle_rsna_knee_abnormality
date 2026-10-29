"""Score ROI predictions without ground-truth boxes, run from src/.

    python -m roi.evaluate round1_pred round2_pred            # validation studies
    python -m roi.evaluate round2_pred --split all

Every sagittal and coronal series has exactly one ROI per class, so each
prediction file has one box per (series, class). Per class:
  - failures: share of boxes below the 0.5 peak-probability threshold;
  - confidence: mean per-box confidence (`roi.consensus.add_confidence`);
  - support: share of the study's other series whose box intersects;
  - iou_other / iou_same: mean best patient-space IoU with a box of the other
    plane / of another series of the same plane;
plus, for the paired classes, the medial / lateral side (against the DICOM
Laterality tag) and the meniscus-inside-its-compartment consistency.
Scored on the validation studies (radiologist-labelled + classifier holdout) by
default, the same series for every file, so rounds compare directly.
"""

import argparse
import os

import pandas as pd

from roi.consensus import add_confidence
from roi.extract import ROI_CLASSES
from roi.labels import LABEL_DIR
from roi.laterality import check, pair_consistency


def evaluate(pred):
    pred = add_confidence(pred)
    t = pred.groupby("class").agg(
        failures=("conf", lambda c: float((c < 0.5).mean())), confidence=("confidence", "mean"),
        support=("support", "mean"), iou_other=("iou_other_plane", "mean"),
        iou_same=("iou_same_plane", "mean"), n=("conf", "size"))
    t = t.reindex([c for c in ROI_CLASSES if c in t.index])
    t.loc["ALL"] = t.drop(columns="n").mean().tolist() + [t.n.sum()]
    side = check(pred)
    pairs, _ = pair_consistency(pred)
    extra = {f"side_{k.lower()}": v for k, (v, _) in side.items()}
    extra.update({f"in_compartment_{k.lower()}": v for k, (v, _) in pairs.items()})
    return t, extra


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+")
    p.add_argument("--split", default="val", choices=["val", "all"])
    a = p.parse_args()
    keep = None
    if a.split == "val":
        from roi.data import load_volumes, split_series

        _, index = load_volumes()
        keep = set(split_series(index)[1].SeriesInstanceUID)
    pd.set_option("display.width", 200)
    for name in a.names:
        pred = pd.read_csv(os.path.join(LABEL_DIR, f"{name}.csv"))
        if keep is not None:
            pred = pred[pred.SeriesInstanceUID.isin(keep)]
        t, extra = evaluate(pred)
        print(f"\n== {name}  ({a.split}: {pred.SeriesInstanceUID.nunique()} series)")
        print(t.round(3).to_string())
        print("   " + "   ".join(f"{k} {v:.3f}" for k, v in extra.items()))


if __name__ == "__main__":
    main()
