"""SKM-TEA pathology annotations (input/skm_tea/{train,val,test}.json) -> graded labels for our findings, run from src/.

    python -m skm.labels          # after skm.prep and ext.boxes --data skm

Annotations are 3D boxes [SI, AP, LR start, SI, AP, LR size] in the raw 0.8 mm slice grid (LR = raw slice
index, increasing towards patient right in all 155 scans; some boxes are drawn in reverse: min / abs).
Medial vs lateral: the box is assigned to the closer of the ROI U-Net's medial / lateral meniscus centres
(output/skm/rois.csv, slab index -> raw index); this placement agrees with knee side + DICOM geometry on
155 / 155 scans. Graded targets, following the competition convention:
  ACL       low-grade sprain 0.25, moderate (partial) tear 0.5, complete tear 1
  menisci   tear 1; myxoid degeneration or extrusion only 0.25
  OA        cartilage lesions (patellar -> PF; femoral / tibial -> medial / lateral by position):
            grade 2B / 3 over >= 10 mm 1; 2B / 3 smaller, or 2A over >= 10 mm 0.5; otherwise 0.25
  Effusion  annotated 1
Scans not inspected by the readers are NaN; MCL, synovitis, Baker's, contusion, fracture: NaN.
Also adds mm_z / lm_z (raw slice index of the meniscus centres) to output/skm/series.csv.
Writes output/skm/labels_graded.csv.
"""

import json
import os

import numpy as np
import pandas as pd

from params import INPUT_FOLDER, OUTPUT_FOLDER

SKM_DIR = os.path.join(OUTPUT_FOLDER, "skm")
FINDINGS = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion"]


def grade(category, tissue, side_comp, surf):
    if category.startswith("Ligament Tear") and tissue == "ACL":
        return "ACL", 1.0 if "Complete" in category else 0.5 if "Moderate" in category else 0.25
    if category.startswith("Meniscal Tear"):
        f = "Medial Meniscus" if side_comp == "medial" else "Lateral Meniscus"
        return f, 0.25 if ("Myxoid" in category or "Extrusion" in category) else 1.0
    if category.startswith("Cartilage Lesion"):
        f = "PF OA" if tissue == "Patellar Cartilage" else ("Medial OA" if side_comp == "medial" else "Lateral OA")
        deep, big = ("(2B)" in category or "(3)" in category), surf >= 10
        return f, 1.0 if (deep and big) else 0.5 if (deep or ("(2A)" in category and big)) else 0.25
    if category == "Effusion":
        return "Effusion", 1.0
    return None, None


def main():
    s = pd.read_csv(os.path.join(SKM_DIR, "series.csv"))
    rois = pd.read_csv(os.path.join(SKM_DIR, "rois.csv"))
    men = rois[rois["class"].isin(["Medial Meniscus", "Lateral Meniscus"])]
    zc = ((men.z0 + men.z1) / 2).groupby([men.SeriesInstanceUID, men["class"]]).first().unstack()
    s = s.set_index("SeriesInstanceUID")
    for c, k in [("Medial Meniscus", "mm"), ("Lateral Meniscus", "lm")]:
        s[f"{k}_z"] = zc[c].reindex(s.index) * s.slab + (s.slab - 1) / 2       # slab index -> raw slice index
    s = s.reset_index()
    s.to_csv(os.path.join(SKM_DIR, "series.csv"), index=False)
    agree = ((s.mm_z < s.lm_z) == s.side.eq("R")).mean()      # right knee: medial towards patient left = lower index
    print(f"medial / lateral placement agrees with side + geometry: {agree:.3f}")
    s = s.set_index("scan_id")

    sp = {k: json.load(open(os.path.join(INPUT_FOLDER, "skm_tea", f"{k}.json"))) for k in ["train", "val", "test"]}
    catn = {c["id"]: c["name"] for c in sp["train"]["categories"]}
    tis = {t["id"]: t["name"] for t in sp["train"]["tissues"]}
    imgs = pd.concat([pd.DataFrame(d["images"]).assign(split=k) for k, d in sp.items()])
    ann = pd.concat([pd.DataFrame(d["annotations"]).assign(split=k) for k, d in sp.items()])
    ann = ann.merge(imgs[["id", "split", "scan_id"]].rename(columns={"id": "image_id"}), on=["image_id", "split"])
    ann["category"] = ann.category_id.map(catn)
    ann["tissue"] = ann.tissue_id.map(tis)
    b = np.stack(ann.bbox.values).astype(float)
    lo, ext = np.minimum(b[:, :3], b[:, :3] + b[:, 3:]), np.abs(b[:, 3:])        # boxes drawn in reverse
    ann["lr"] = lo[:, 2] + ext[:, 2] / 2
    ann["surf_mm"] = np.maximum(ext[:, 1] * 0.3125, ext[:, 2] * 0.8)               # extent along the surface (AP / LR)
    ann["side_comp"] = ["medial" if abs(lr - s.loc[sc, "mm_z"]) < abs(lr - s.loc[sc, "lm_z"]) else "lateral"
                        for lr, sc in zip(ann.lr, ann.scan_id)]
    gv = [grade(*x) for x in zip(ann.category, ann.tissue, ann.side_comp, ann.surf_mm)]
    ann["finding"], ann["value"] = [x[0] for x in gv], [x[1] for x in gv]
    ann = ann.dropna(subset=["finding"])
    lab = ann.pivot_table(index="scan_id", columns="finding", values="value", aggfunc="max").reindex(
        index=imgs.scan_id.values, columns=FINDINGS).fillna(0.0)
    insp = imgs.set_index("scan_id").loc[lab.index, "inspected"].values
    lab.loc[~insp.astype(bool)] = np.nan                                            # not inspected: unlabelled
    lab.index.name = "scan_id"
    lab["StudyInstanceUID"] = "SKM_" + lab.index
    lab["split"] = imgs.set_index("scan_id").loc[lab.index, "split"].values
    for c in ["MCL", "Synovitis", "Baker's", "Contusion", "Fracture"]:
        lab[c] = np.nan
    lab.to_csv(os.path.join(SKM_DIR, "labels_graded.csv"))
    print(f"{len(lab)} scans ({int(lab['ACL'].notna().sum())} labelled)")
    print(lab[FINDINGS].describe().T[["count", "mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
