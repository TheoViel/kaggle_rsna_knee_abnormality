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


OUTSIDE = False          # --outside: map locations outside every box to the nearest crop(s), at half value
FIBULA = False           # --fibula: the ninth crop (fracture.fibula) takes fibular-head / fibula fractures
BONES = False            # --bones: + the 4 bone crops (fracture.bones): distal femur / proximal tibia, medial / lateral
OC = 1.0                 # --oc: factor for osteochondral fractures
EXTRA3 = False           # --extra3: + 3 crops (fracture.extra): whole patella, tibial tuberosity, fibular neck / shaft
EXTRA_ROIS = ["Patella", "Tibial Tuberosity", "Fibula Shaft"]


def extra_rois(loc):
    """--extra3: the extra crop(s) of one reported location, with their factor."""
    s, r, d = loc["structure"], loc["region"], loc.get("detail", "").lower()
    out = []
    if s in ("patella", "trochlea") or "patell" in d:
        out.append(("Patella", 1.0))
    if (s == "proximal tibia (other)" and r in ("anterior", "central", "unk")) or any(k in d for k in ("tuberosit", "tubercle", "anterior tibia", "patellar tendon")):
        out.append(("Tibial Tuberosity", 1.0))
    if "fibula" in d or "fibular" in d or s == "fibular head":
        out.append(("Fibula Shaft", 0.5 if s == "fibular head" and not any(k in d for k in ("neck", "shaft")) else 1.0))
    return out
BONE_ROIS = ["Medial Femur", "Lateral Femur", "Medial Tibia", "Lateral Tibia"]


def bone_rois(loc):
    """--bones: the bone crop(s) of one reported location, with their factor (added to rois())."""
    s, r = loc["structure"], loc["region"]
    side = r.capitalize() if r in SIDE else None
    if s == "femoral condyle" or s == "distal femur (other)":
        return [(f"{side} Femur", 1.0)] if side else [("Medial Femur", 0.5), ("Lateral Femur", 0.5)]
    if s == "tibial plateau" or s == "proximal tibia (other)":
        return [(f"{side} Tibia", 1.0)] if side else [("Medial Tibia", 0.5), ("Lateral Tibia", 0.5)]
    if s == "tibial intercondylar eminence":                 # central: the inner edge of both tibia crops
        return [("Medial Tibia", 0.5), ("Lateral Tibia", 0.5)]
    return []


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
    if FIBULA and (s == "fibular head" or ("fibula" in loc.get("detail", "").lower() and r in ("lateral", "unk"))):
        return ["Fibula"], 1.0
    if s == "fibular head":               # at the lower edge of the coronal lateral OA crop only
        return ["Lateral OA"], 0.5
    if OUTSIDE:   # outside every box: the nearest crop(s), where secondary signs (edema, effusion) can show
        if s in ("other", "distal femur (other)") and r == "lateral":   # fibula (level unknown), lateral femur
            return ["Lateral OA"], 0.5
        if s == "proximal tibia (other)" and r == "anterior":           # tibial tuberosity region
            return ["PF OA"], 0.5
        if s == "proximal tibia (other)":                                # side unknown
            return SIDE["medial"] + SIDE["lateral"], 0.5
    return [], 0.0


def unknown_rois():
    """--outside: a fracture with no location at all -> both compartments and PF, at half value."""
    return SIDE["medial"] + SIDE["lateral"] + ["PF OA"], 0.5


def main():
    import argparse

    global ACUITY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    global OUTSIDE, FIBULA, BONES, EXTRA3
    ap.add_argument("--version", default="v2", choices=list(ACUITY_VERSIONS))
    ap.add_argument("--neg_zero", action="store_true", help="studies outside the extraction (Claude < 0.3): 0, not the Claude label")
    ap.add_argument("--outside", action="store_true", help="map locations outside every box to the nearest crop(s) (OUTSIDE)")
    ap.add_argument("--fibula", action="store_true", help="add the fibular-head ROI (FIBULA): 9 per-ROI targets")
    ap.add_argument("--extra3", action="store_true", help="add the 3 extra crops (EXTRA3, fracture.extra): 16 per-ROI targets")
    ap.add_argument("--bones", action="store_true", help="add the 4 bone ROIs (BONES, fracture.bones): 13 per-ROI targets")
    ap.add_argument("--oc", type=float, default=1.0,
                    help="factor for locations whose detail says 'osteochondral' (3 / 4 radiologist negatives on the 58)")
    ap.add_argument("--trauma", type=float, default=0.0,
                    help="pivot-shift rule: studies with a traumatic contusion (contusion_locations.json), an ACL tear "
                         "(Claude >= 0.5) and no report fracture -> this value at the contusion sites' ROIs (radiologists "
                         "of the 58 mark 4 / 8 of them fractured, 0 / 30 other studies without a report fracture)")
    a = ap.parse_args()
    ACUITY = ACUITY_VERSIONS[a.version]
    global OC
    OUTSIDE, FIBULA, BONES, OC, EXTRA3 = a.outside, a.fibula, a.bones, a.oc, a.extra3
    suffix = ("" if a.version == "v1" else f"_{a.version}") + ("_z" if a.neg_zero else "") + ("_o" if a.outside else "") \
        + ("_fib" if a.fibula else "") + ("_bones" if a.bones else "") + ("_x3" if a.extra3 else "") + (f"_tr{int(a.trauma * 100)}" if a.trauma else "") \
        + (f"_oc{int(a.oc * 100)}" if a.oc != 1.0 else "")
    rows = json.load(open(os.path.join(LOC, "fracture_locations.json")))["studies"]
    claude = pd.read_csv(CLAUDE_LABELS_CSV).set_index("StudyInstanceUID")["Fracture"]
    roi_classes = ROI_CLASSES + (["Fibula"] if FIBULA else []) + (BONE_ROIS if BONES else []) + (EXTRA_ROIS if EXTRA3 else [])
    cols = [f"{c}|fracture" for c in roi_classes]
    base = claude * 0.0 if a.neg_zero else claude
    out = pd.DataFrame(np.repeat(base.values[:, None], len(cols), 1), index=claude.index, columns=cols)
    out["Fracture"] = base
    n_mapped = n_unmapped = 0
    for r in rows:
        uid = r["StudyInstanceUID"]
        if r["fracture"] == "no":
            out.loc[uid, cols] = 0.0
            out.loc[uid, "Fracture"] = 0.0
            continue
        t = dict.fromkeys(roi_classes, 0.0)
        hit = False
        locs = r["locations"]
        if OUTSIDE and locs == "unk":
            locs = [dict(structure="unknown", region="unk", certainty="possible" if r["fracture"] == "possible" else "definite")]
        if isinstance(locs, list):
            for loc in locs:
                rs, f = unknown_rois() if loc["structure"] == "unknown" else rois(loc)
                oc = OC if "osteochondral" in loc.get("detail", "").lower() else 1.0
                v = CERTAINTY[loc["certainty"]] * ACUITY.get(r["acuity"] or "unk", 0.75) * f * oc
                for c in rs:
                    t[c] = max(t[c], v)
                    hit = True
                if EXTRA3 and loc["structure"] != "unknown":
                    for c, fx in extra_rois(loc):
                        t[c] = max(t[c], CERTAINTY[loc["certainty"]] * ACUITY.get(r["acuity"] or "unk", 0.75) * fx * oc)
                        hit = True
                if BONES and loc["structure"] != "unknown":
                    for c, fb in bone_rois(loc):
                        t[c] = max(t[c], CERTAINTY[loc["certainty"]] * ACUITY.get(r["acuity"] or "unk", 0.75) * fb * oc)
                        hit = True
        study = max([CERTAINTY[l["certainty"]] for l in r["locations"]] if isinstance(r["locations"], list) and r["locations"] else [0.5]) \
            * ACUITY.get(r["acuity"] or "unk", 0.75)
        out.loc[uid, "Fracture"] = study
        if hit:
            out.loc[uid, cols] = [t[c] for c in roi_classes]
            n_mapped += 1
        else:
            out.loc[uid, cols] = np.nan
            n_unmapped += 1
    if a.trauma:
        from fracture.cont_labels import map_site

        acl = pd.read_csv(CLAUDE_LABELS_CSV).set_index("StudyInstanceUID")["ACL"]
        frac = {r["StudyInstanceUID"]: r["fracture"] for r in rows}
        n_tr = 0
        for c in json.load(open(os.path.join(LOC, "contusion_locations.json")))["studies"]:
            uid = c["StudyInstanceUID"]
            sites = [x for x in (c["sites"] if isinstance(c["sites"], list) else []) if x["type"] == "traumatic"]
            if not sites or frac.get(uid) in ("yes", "possible") or not acl.get(uid, 0) >= 0.5:
                continue
            t = dict.fromkeys(roi_classes, 0.0)
            for x in sites:
                rs, f = map_site(x)
                hits = [(r_, f) for r_ in rs] + (bone_rois(x) if BONES else [])
                for r_, f_ in hits:
                    if r_ in t:
                        t[r_] = max(t[r_], a.trauma * CERTAINTY.get(x["certainty"], 0.5) * f_)
            if max(t.values()) > 0:
                out.loc[uid, cols] = [t[c_] for c_ in roi_classes]
                out.loc[uid, "Fracture"] = a.trauma
                n_tr += 1
        print(f"--trauma {a.trauma}: {n_tr} studies (traumatic contusion + ACL tear, no report fracture) get fracture targets")
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
