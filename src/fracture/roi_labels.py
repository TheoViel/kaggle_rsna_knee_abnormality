"""Report fracture locations (output/locations/fracture_locations.json) -> per-ROI fracture targets, run from src/.

    python -m fracture.roi_labels

Each reported location (structure, region) is mapped to the ROI crops that contain it:
  tibial plateau, femoral condyle   medial -> Medial OA, Medial Meniscus; lateral -> Lateral OA, Lateral Meniscus;
                                    no side -> all four at half value
  tibial intercondylar eminence     ACL (the notch and the tibial ACL attachment)
  patella, trochlea                 PF OA
  proximal tibia (other)            medial / lateral rim (e.g. Segond) -> that compartment's OA + meniscus
  distal femur (other)              medial (epicondyle, MCL origin) -> MCL
  fibular head                      Lateral OA at half value (lower edge of the coronal crop only)
  tibial tuberosity, other: no ROI (outside every box)
Value of a location (graded; convention: "an acute cortical break or fracture line"): certainty definite 1 /
possible 0.5, times acuity (--version v2, default): acute 1 / insufficiency 0.75 / unk 0.75 / old 0. On the 58,
the radiologists marked 2 of 3 report-insufficiency fractures positive (a fracture line), and old fractures are not
acute. v1 (first runs): insufficiency 0.5, old 0.25. An ROI's target is the max over the study's
locations mapped to it; the study's other ROIs are 0. Studies whose fracture has no mappable location get
no ROI targets (NaN). Reports read as "no fracture" -> all ROIs 0. Studies outside the extraction (Claude
fracture label < 0.3) -> every ROI = their Claude label (low).
Writes output/locations/fracture_roi_targets[_v2].csv: StudyInstanceUID x "<ROI>|fracture", plus "Fracture" (study),
and the training label file output/crops/frac_roi[_v2].csv.
"""

import json
import os

import numpy as np
import pandas as pd

from params import CLAUDE_LABELS_CSV, OUTPUT_FOLDER
from roi.extract import ROI_CLASSES

LOC = os.path.join(OUTPUT_FOLDER, "locations")
SIDE = {"medial": ["Medial OA", "Medial Meniscus"], "lateral": ["Lateral OA", "Lateral Meniscus"]}
CERTAINTY = {"definite": 1.0, "possible": 0.5}
ACUITY_VERSIONS = {"v1": {"acute": 1.0, "unk": 0.75, "insufficiency": 0.5, "old": 0.25},
                   "v2": {"acute": 1.0, "unk": 0.75, "insufficiency": 0.75, "old": 0.0}}
ACUITY = ACUITY_VERSIONS["v2"]


def rois(loc):
    """(ROI list, factor) for one reported location."""
    s, r = loc["structure"], loc["region"]
    if s in ("tibial plateau", "femoral condyle"):
        return (SIDE[r], 1.0) if r in SIDE else (SIDE["medial"] + SIDE["lateral"], 0.5)
    if s == "tibial intercondylar eminence":
        return ["ACL"], 1.0
    if s in ("patella", "trochlea"):
        return ["PF OA"], 1.0
    if s == "proximal tibia (other)" and r in SIDE:
        return SIDE[r], 1.0
    if s == "distal femur (other)" and r == "medial":
        return ["MCL"], 1.0
    if s == "fibular head":               # at the lower edge of the coronal lateral OA crop only
        return ["Lateral OA"], 0.5
    return [], 0.0


def main():
    import argparse

    global ACUITY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v2", choices=list(ACUITY_VERSIONS))
    a = ap.parse_args()
    ACUITY = ACUITY_VERSIONS[a.version]
    suffix = "" if a.version == "v1" else f"_{a.version}"
    rows = json.load(open(os.path.join(LOC, "fracture_locations.json")))["studies"]
    claude = pd.read_csv(CLAUDE_LABELS_CSV).set_index("StudyInstanceUID")["Fracture"]
    cols = [f"{c}|fracture" for c in ROI_CLASSES]
    out = pd.DataFrame(np.repeat(claude.values[:, None], len(cols), 1), index=claude.index, columns=cols)
    out["Fracture"] = claude
    n_mapped = n_unmapped = 0
    for r in rows:
        uid = r["StudyInstanceUID"]
        if r["fracture"] == "no":
            out.loc[uid, cols] = 0.0
            out.loc[uid, "Fracture"] = 0.0
            continue
        t = dict.fromkeys(ROI_CLASSES, 0.0)
        hit = False
        if isinstance(r["locations"], list):
            for loc in r["locations"]:
                rs, f = rois(loc)
                v = CERTAINTY[loc["certainty"]] * ACUITY.get(r["acuity"] or "unk", 0.75) * f
                for c in rs:
                    t[c] = max(t[c], v)
                    hit = True
        study = max([CERTAINTY[l["certainty"]] for l in r["locations"]] if isinstance(r["locations"], list) and r["locations"] else [0.5]) \
            * ACUITY.get(r["acuity"] or "unk", 0.75)
        out.loc[uid, "Fracture"] = study
        if hit:
            out.loc[uid, cols] = [t[c] for c in ROI_CLASSES]
            n_mapped += 1
        else:
            out.loc[uid, cols] = np.nan
            n_unmapped += 1
    out.rename_axis("StudyInstanceUID").to_csv(os.path.join(LOC, f"fracture_roi_targets{suffix}.csv"))
    # Training label file for main_crops --labels pl:frac_roi{suffix}
    out.drop(columns=["Fracture"]).rename_axis("StudyInstanceUID").to_csv(os.path.join(OUTPUT_FOLDER, "crops", f"frac_roi{suffix}.csv"))
    pos = [r for r in rows if r["fracture"] != "no"]
    print(f"{len(pos)} studies with a (possible) fracture: {n_mapped} mapped to >= 1 ROI, {n_unmapped} not "
          f"(no location inside a box)")
    ex = out.loc[[r["StudyInstanceUID"] for r in pos]]
    print("positive ROI targets (>= 0.5) per ROI:", (ex[cols] >= 0.5).sum().rename(lambda c: c.split("|")[0]).to_dict())
    print("study target distribution:", ex["Fracture"].round(3).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
