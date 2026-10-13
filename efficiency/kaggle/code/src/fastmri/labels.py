"""fastMRI+ boxes (input/fastmri/knee.csv) -> graded labels for our findings, study and per ROI; run from src/.

    python -m fastmri.labels          # after fastmri.prep and ext.boxes --data fastmri

fastMRI+ annotates each volume completely (boxes on slices + a label), so a finding without a box is negative,
and volumes without any box are normal. Volumes are coronal only. Each box is assigned to the ROI box of our
U-Net it overlaps most on its slice (both in the vertically flipped frame of fastmri.prep), among the ROIs its
label can belong to: menisci for meniscus tears, the three compartments for cartilage / subchondral findings,
any ROI for bone findings. This resolves medial / lateral by anatomy, without orientation conventions.

Graded targets (competition convention; 1 = convention-positive, intermediate grades 0.25 / 0.5):
  ACL               high-grade sprain 1, low-grade 0.25
  MCL               high-grade sprain 1, low-moderate 0.25
  menisci           tear / displaced meniscal tissue (by ROI): >= 2 boxes on that meniscus 1, a single box 0.5
  OA (medial / lateral; PF from the PF box)  cartilage full-thickness loss 1 if >= 10 mm wide, else 0.5;
                    partial-thickness 0.25 (>= 10 mm: 0.5)
  Effusion          joint effusion 1 (study level)
  Contusion         bone fracture / contusion / dislocation 1 (the label does not separate them)
  Fracture          the same boxes at 0.5 (per ROI: "<ROI>|fracture")
  Baker's           periarticular cyst overlapping the Baker's box 0.5 (the label covers other cysts too)
Writes output/fastmri/labels_graded.csv (StudyInstanceUID x findings), labels_roi.csv (per-ROI fracture, 0.5) and
labels_bone9.csv: per-ROI bone injury over the 9 ROIs of the fracture models (+ the fibular head, fracture.fibula),
1 where a fracture / contusion / dislocation box falls, else 0 (for pretraining: the label does not separate them).
"""

import os

import numpy as np
import pandas as pd

from params import INPUT_FOLDER, OUTPUT_FOLDER
from roi.extract import ROI_CLASSES

FM = os.path.join(OUTPUT_FOLDER, "fastmri")
PX_MM = 140.0 / 320            # ~0.44 mm per pixel (reconstruction_esc: the central 140 mm of the 280 mm FOV)
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture"]
CANDIDATES = {"men": ["Medial Meniscus", "Lateral Meniscus"], "comp": ["Medial OA", "Lateral OA", "PF OA"],
              "any": list(ROI_CLASSES), "any9": list(ROI_CLASSES) + ["Fibula"]}


def overlap(box, roi):
    """Fraction of a fastMRI+ box (x0, y0, x1, y1) inside an ROI box's rows / columns."""
    ix = max(0, min(box[2], roi["x1"]) - max(box[0], roi["x0"]))
    iy = max(0, min(box[3], roi["y1"]) - max(box[1], roi["y0"]))
    return ix * iy / max((box[2] - box[0]) * (box[3] - box[1]), 1)


def assign(r, rois, kind):
    """The ROI (among `kind`) a box belongs to: max overlap on its slice; else the nearest centre."""
    box = (r.x, r.y, r.x + r.width, r.y + r.height)
    cands = [c for c in CANDIDATES[kind] if c in rois]
    on_slice = [c for c in cands if rois[c]["z0"] - 1 <= r.slice <= rois[c]["z1"] + 1]
    pool = on_slice or cands
    best = max(pool, key=lambda c: overlap(box, rois[c]), default=None)
    if best is not None and overlap(box, rois[best]) > 0:
        return best
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(pool, key=lambda c: (cx - (rois[c]["x0"] + rois[c]["x1"]) / 2) ** 2
               + (cy - (rois[c]["y0"] + rois[c]["y1"]) / 2) ** 2, default=None)


def main():
    s = pd.read_csv(os.path.join(FM, "series.csv"))
    s = s[s.status == "ok"]
    rois = {}
    rb = pd.read_csv(os.path.join(FM, "rois.csv"))
    for r in rb.to_dict("records"):
        rois.setdefault(r["StudyInstanceUID"], {})[r["class"]] = {k: r[k] for k in ["z0", "z1", "y0", "y1", "x0", "x1"]}
    k = pd.read_csv(os.path.join(INPUT_FOLDER, "fastmri", "knee.csv"))
    k = k[(k.study_level == "No") & k.x.notna()].copy()
    k["label"] = k.label.str.strip()                          # e.g. "Joint Effusion " in the csv
    k["StudyInstanceUID"] = "FASTMRI_" + k.file
    k = k[k.StudyInstanceUID.isin(s.StudyInstanceUID)]
    lab = pd.DataFrame(0.0, index=pd.Index(s.StudyInstanceUID, name="StudyInstanceUID"), columns=FINDINGS)
    lab["Synovitis"] = np.nan                                   # not annotated
    roi_frac = pd.DataFrame(0.0, index=lab.index, columns=[f"{c}|fracture" for c in ROI_CLASSES])
    bone9 = pd.DataFrame(0.0, index=lab.index, columns=[f"{c}|fracture" for c in ROI_CLASSES + ["Fibula"]])
    from fracture.fibula import fibula_box
    shape = {r.StudyInstanceUID: r for r in s.itertuples()}
    rois9 = {}
    for uid, bx in rois.items():
        r = shape.get(uid)
        fb = fibula_box(bx, "Coronal", r.rows, r.cols, r.n_slices) if r is not None else None
        rois9[uid] = {**bx, **({"Fibula": fb} if fb else {})}
    unassigned = 0
    men_boxes = {}                                              # (study, meniscus) -> number of tear boxes
    for r in k.itertuples():
        uid, L, rb_ = r.StudyInstanceUID, r.label, rois.get(r.StudyInstanceUID, {})
        wide = r.width * PX_MM >= 10

        def up(c, v):
            lab.loc[uid, c] = max(lab.loc[uid, c], v)
        if L.startswith("Ligament - ACL"):
            up("ACL", 1.0 if "High" in L else 0.25)
        elif L.startswith("Ligament - MCL"):
            up("MCL", 1.0 if "High" in L else 0.25)
        elif L in ("Meniscus Tear", "Displaced Meniscal Tissue"):
            c = assign(r, rb_, "men")
            if c:
                men_boxes[(uid, c)] = men_boxes.get((uid, c), 0) + 1
            else:
                unassigned += 1
        elif L.startswith("Cartilage"):
            c = assign(r, rb_, "comp")
            if c:
                up(c, (1.0 if wide else 0.5) if "Full" in L else (0.5 if wide else 0.25))
            else:
                unassigned += 1
        elif L == "Joint Effusion":
            up("Effusion", 1.0)
        elif L.startswith("Bone-Fracture"):
            up("Contusion", 1.0)
            up("Fracture", 0.5)
            c = assign(r, rb_, "any")
            if c:
                roi_frac.loc[uid, f"{c}|fracture"] = max(roi_frac.loc[uid, f"{c}|fracture"], 0.5)
            c9 = assign(r, rois9.get(uid, {}), "any9")
            if c9:
                bone9.loc[uid, f"{c9}|fracture"] = 1.0
        elif L == "Periarticular cysts":
            if "Baker's" in rb_ and overlap((r.x, r.y, r.x + r.width, r.y + r.height), rb_["Baker's"]) > 0.3:
                up("Baker's", 0.5)
    # A tear is annotated on every slice it shows on: >= 2 boxes on a meniscus -> 1; a single box -> 0.5
    # (on a volume with many boxes on the other meniscus, a lone box is often a stray assignment).
    for (uid, c), n in men_boxes.items():
        lab.loc[uid, c] = 1.0 if n >= 2 else 0.5
    lab.to_csv(os.path.join(FM, "labels_graded.csv"))
    roi_frac.to_csv(os.path.join(FM, "labels_roi.csv"))
    bone9.to_csv(os.path.join(FM, "labels_bone9.csv"))
    print("bone injury per ROI (9):", (bone9 > 0).sum().rename(lambda c: c.split("|")[0]).to_dict())
    print(f"{len(lab)} volumes, {k.StudyInstanceUID.nunique()} with boxes, {unassigned} boxes without a ROI")
    print(pd.DataFrame({"mean": lab.mean().round(3), "positive (>= 0.5)": (lab >= 0.5).sum(), "> 0": (lab > 0).sum()}).to_string())


if __name__ == "__main__":
    main()
