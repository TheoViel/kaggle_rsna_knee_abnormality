"""Report bone-marrow-edema sites (output/locations/contusion_locations.json) -> per-ROI contusion targets, run from src/.

    python -m fracture.cont_labels --weights trauma           # writes output/crops/cont_roi_<weights>.csv

Convention: "Contusion: bone marrow edema-like signal without a fracture line". Each report site (contusion
extraction: structure, region, type, extent, certainty) is mapped to the ROI crops containing it with the fracture
mapping (fracture.roi_labels.rois, 9 ROIs: the fibular head has its own crop). Value of a site: certainty (definite 1,
possible 0.5) x the weight of its type (TYPE_WEIGHTS[--weights]). An ROI's target is the max over the study's sites
mapped to it; the study's other ROIs are 0. Reports with no edema -> every ROI 0; edema without a usable location ->
NaN (left out). Studies outside the extraction (Claude contusion < 0.3) -> every ROI = their Claude label.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from fracture import roi_labels
from params import CLAUDE_LABELS_CSV, OUTPUT_FOLDER
from roi.extract import ROI_CLASSES

LOC = os.path.join(OUTPUT_FOLDER, "locations")
ROIS9 = ROI_CLASSES + ["Fibula"]
CERTAINTY = {"definite": 1.0, "possible": 0.5}
TYPE_WEIGHTS = {
    "any": {"traumatic": 1.0, "unk": 1.0, "degenerative": 1.0, "fracture": 1.0, "other": 1.0},
    "trauma": {"traumatic": 1.0, "unk": 1.0, "degenerative": 0.5, "fracture": 0.5, "other": 0.25},
    "strict": {"traumatic": 1.0, "unk": 0.75, "degenerative": 0.0, "fracture": 0.0, "other": 0.0},
    # trauma + extent: on the 58, 3 of the 5 studies whose traumatic edema is only "mild" are radiologist negatives
    "trauma_ext": {"traumatic": 1.0, "unk": 1.0, "degenerative": 0.5, "fracture": 0.5, "other": 0.25, "_mild": 0.6},
}


def map_site(s):
    """ROI crops of an edema site: the fracture mapping; a femur / tibia site without a side (or an unnamed bone)
    -> both tibiofemoral compartments (OA + meniscus crops) at half value; lateral / anterior outside the boxes as
    fracture.roi_labels --outside does."""
    rs, f = roi_labels.rois(s)
    if rs:
        return rs, f
    if s["structure"] in ("proximal tibia (other)", "distal femur (other)", "other") and s["region"] in ("unk", "central", "posterior"):
        return roi_labels.SIDE["medial"] + roi_labels.SIDE["lateral"], 0.5
    roi_labels.OUTSIDE = True
    try:
        return roi_labels.rois(s)
    finally:
        roi_labels.OUTSIDE = False


def build(weights, rows, claude):
    roi_labels.FIBULA = True
    cols = [f"{c}|contusion" for c in ROIS9]
    out = pd.DataFrame(np.repeat(claude.values[:, None], len(cols), 1), index=claude.index, columns=cols)
    stats = dict(mapped=0, unmapped=0, no=0)
    w = TYPE_WEIGHTS[weights] if isinstance(weights, str) else weights
    for r in rows:
        uid = r["StudyInstanceUID"]
        if r["edema"] == "no":
            out.loc[uid, cols] = 0.0
            stats["no"] += 1
            continue
        t, hit = dict.fromkeys(ROIS9, 0.0), False
        if isinstance(r["sites"], list):
            for s in r["sites"]:
                rs, f = map_site(s)
                v = CERTAINTY.get(s.get("certainty"), 0.5) * w.get(s.get("type", "unk"), w["unk"]) * f
                if s.get("extent") == "mild":
                    v *= w.get("_mild", 1.0)
                for c in rs:
                    t[c] = max(t[c], v)
                    hit = True
        if hit:
            out.loc[uid, cols] = [t[c] for c in ROIS9]
            stats["mapped"] += 1
        else:
            out.loc[uid, cols] = np.nan
            stats["unmapped"] += 1
    return out, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="trauma", choices=list(TYPE_WEIGHTS))
    a = p.parse_args()
    rows = json.load(open(os.path.join(LOC, "contusion_locations.json")))["studies"]
    claude = pd.read_csv(CLAUDE_LABELS_CSV).set_index("StudyInstanceUID")["Contusion"]
    out, st = build(a.weights, rows, claude)
    out.rename_axis("StudyInstanceUID").to_csv(os.path.join(OUTPUT_FOLDER, "crops", f"cont_roi_{a.weights}.csv"))
    ex = out.loc[[r["StudyInstanceUID"] for r in rows if r["edema"] != "no"]]
    print(f"--weights {a.weights}: {st}; positive ROI targets (>= 0.5):",
          (ex >= 0.5).sum().rename(lambda c: c.split("|")[0]).to_dict())


if __name__ == "__main__":
    main()
