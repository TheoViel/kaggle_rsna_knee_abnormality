"""MOAKS (OAI semi-quantitative MRI readings) -> labels for our findings, per knee, run from src/.

    python -m oai.labels --visit 00             # v2 rules below (superseded) -> labels_moaks00.csv
    python -m oai.labels --graded               # graded convention labels (in use) -> labels_moaks00_v4.csv
    python -m oai.labels --aux                  # 25 MOAKS gradings for pretraining -> aux_moaks00.csv

One knee = one "study" `OAI_<participant id>_<RIGHT|LEFT>`. Several reading projects scored
some knees; their labels are averaged (soft labels).

Compartments follow the MOAKS documentation (kMRI_SQ_MOAKS_BICL_Descrip):
  medial tibio-femoral   FMC FMP TMA TMC TMP       (osteophytes: FMC FMP TM)
  lateral tibio-femoral  FLC FLP TLA TLC TLP       (osteophytes: FLC FLP TL)
  patello-femoral        FMA FLA PM PL             (osteophytes: FMA FLA PS PI PM PL)

Cartilage scores are "size.full_thickness" (size 0-3 = none / <10% / 10-75% / >75% of the
subregion; full thickness 0-3 alike). Osteophytes are only scored for ~20% of the knees, so the
OA definition (in the spirit of Hunter et al. 2011, MRI OA) leans on cartilage:
  OA = 1    if full-thickness loss and a lesion covering >= 10% of a subregion of the compartment,
            or a medium / large osteophyte (grade >= 2) where osteophytes were scored,
            or a >= 10% lesion with a large bone marrow lesion (grade >= 2) or meniscal maceration
            (tibio-femoral compartments);
  OA = 0.5  if a >= 10% lesion with any bone marrow lesion or a small osteophyte (borderline);
  OA = 0    otherwise.
Meniscal tear (medial / lateral): any radial / horizontal / vertical / complex tear or maceration
in the anterior horn, body or posterior horn, or a posterior root tear -> 1; signal abnormality
only -> 0. ACL: complete tear 1, partial tear 0.5. Effusion (whole-knee effusion-synovitis) and
Hoffa synovitis: grade 0 -> 0, 1 (mild) -> 0.25, >= 2 -> 1. Baker's: popliteal cyst present.
MCL, contusion and fracture have no MOAKS counterpart (NaN).
"""

import argparse
import os

import numpy as np
import pandas as pd

from params import INPUT_FOLDER, OUTPUT_FOLDER

MOAKS_DIR = os.path.join(INPUT_FOLDER, "oai", "Semi-Quant Scoring_ASCII")
OAI_DIR = os.path.join(OUTPUT_FOLDER, "oai")
COMP = {
    "Medial OA": dict(cart=["FMC", "FMP", "TMA", "TMC", "TMP"], bml=["FMC", "FMP", "TMA", "TMC", "TMP"], ost=["FMC", "FMP", "TM"],
                      men="M"),
    "Lateral OA": dict(cart=["FLC", "FLP", "TLA", "TLC", "TLP"], bml=["FLC", "FLP", "TLA", "TLC", "TLP"], ost=["FLC", "FLP", "TL"],
                       men="L"),
    "PF OA": dict(cart=["FMA", "FLA", "PM", "PL"], bml=["FMA", "FLA", "PM", "PL"], ost=["FMA", "FLA", "PS", "PI", "PM", "PL"], men=None),
}


def code(x):
    """'2.1: 10-75% area, ...' -> 2.1; missing -> NaN."""
    s = str(x).split(":")[0].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def oa_label(r, v, comp):
    c = COMP[comp]
    cart = np.array([code(r.get(f"V{v}MCM{s}")) for s in c["cart"]])
    size, ft = np.floor(cart), np.round((cart - np.floor(cart)) * 10)
    bml = np.array([code(r.get(f"V{v}MBMS{s}")) for s in c["bml"]])
    ost = np.array([code(r.get(f"V{v}MOS{s}")) for s in c["ost"]])
    mac = 0
    if c["men"]:
        mac = max((code(r.get(f"V{v}MMT{c['men']}{p}")) in (6, 7, 8)) for p in "ABP")
    big_lesion = np.nanmax(size, initial=0) >= 2
    full = np.nanmax(ft, initial=0) >= 1
    ost_max = np.nanmax(ost, initial=0) if np.isfinite(ost).any() else np.nan
    bml_max = np.nanmax(bml, initial=0)
    if (full and big_lesion) or (ost_max >= 2) or (big_lesion and (bml_max >= 2 or mac)):
        return 1.0
    if big_lesion and (bml_max >= 1 or ost_max >= 1):
        return 0.5
    if not np.isfinite(cart).any():
        return np.nan
    return 0.0


def meniscus_label(r, v, side):
    t = [code(r.get(f"V{v}MMT{side}{p}")) for p in "ABP"]
    root = code(r.get(f"V{v}MMRT{side}"))
    if not np.isfinite(t).any():
        return np.nan
    return float(any(x in (2, 3, 4, 5, 6, 7, 8) for x in t) or root == 1)


def graded(x, mild=0.25):
    x = code(x)
    return np.nan if not np.isfinite(x) else (0.0 if x == 0 else mild if x == 1 else 1.0)


def knee_labels(r, v):
    acl = code(r.get(f"V{v}MACLTR"))
    pop = code(r.get(f"V{v}MPOPCYS"))
    return {
        "ACL": np.nan if not np.isfinite(acl) else {0: 0.0, 1: 0.5, 2: 1.0}.get(int(acl), np.nan),
        "Medial Meniscus": meniscus_label(r, v, "M"), "Lateral Meniscus": meniscus_label(r, v, "L"),
        "Medial OA": oa_label(r, v, "Medial OA"), "Lateral OA": oa_label(r, v, "Lateral OA"), "PF OA": oa_label(r, v, "PF OA"),
        "Effusion": graded(r.get(f"V{v}MEFFWK")), "Synovitis": graded(r.get(f"V{v}MSYIC")),
        "Baker's": np.nan if not np.isfinite(pop) else float(pop == 1),
    }


def build(visit="00"):
    m = pd.read_csv(os.path.join(MOAKS_DIR, f"kMRI_SQ_MOAKS_BICL{visit}.txt"), sep="|", low_memory=False)
    m["side"] = m.SIDE.astype(str).str.extract(r"(Right|Left)", expand=False).str.upper()
    rows = [dict(StudyInstanceUID=f"OAI_{r['ID']}_{r['side']}", oai_id=str(r["ID"]), side=r["side"], project=r["READPRJ"],
                 **knee_labels(r, visit)) for r in m.to_dict("records")]
    d = pd.DataFrame(rows)
    lab = [c for c in d.columns if c not in ("StudyInstanceUID", "oai_id", "side", "project")]
    out = d.groupby(["StudyInstanceUID", "oai_id", "side"])[lab].mean().reset_index()     # several reading projects -> mean
    for c in ("MCL", "Contusion", "Fracture"):
        out[c] = np.nan
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--visit", default="00")
    p.add_argument("--graded", action="store_true", help="graded convention labels (build_graded) -> labels_moaks<visit>_v4.csv")
    p.add_argument("--aux", action="store_true", help="the 25 MOAKS pretraining targets (build_aux) -> aux_moaks<visit>.csv")
    a = p.parse_args()
    os.makedirs(OAI_DIR, exist_ok=True)
    if a.graded or a.aux:
        out = build_graded(a.visit) if a.graded else build_aux(a.visit)
        path = os.path.join(OAI_DIR, f"labels_moaks{a.visit}_v4.csv" if a.graded else f"aux_moaks{a.visit}.csv")
        out.to_csv(path, index=False)
        print(f"-> {path}: {len(out)} knees")
        return
    out = build(a.visit)
    path = os.path.join(OAI_DIR, f"labels_moaks{a.visit}.csv")
    out.to_csv(path, index=False)
    print(f"-> {path}: {len(out)} knees")
    lab = [c for c in out.columns if c not in ("StudyInstanceUID", "oai_id", "side")]
    print(pd.DataFrame({"labelled": out[lab].notna().sum(), "mean (prevalence)": out[lab].mean().round(3),
                        "positive (>= 0.5)": (out[lab] >= 0.5).sum()}).to_string())



# --------------------------------------------- MOAKS auxiliary targets (pretraining)

def aux_targets(r, v="00"):
    """Fine-grained MOAKS gradings, each scaled to [0, 1] and tied to one of our ROI crops
    (name "<ROI>|<feature>"); for multi-task pretraining on OAI (crops.train.crop_index splits on "|")."""
    out = {}
    for comp, roi in (("Medial OA", "Medial OA"), ("Lateral OA", "Lateral OA"), ("PF OA", "PF OA")):
        c = COMP[comp]
        cart = np.array([code(r.get(f"V{v}MCM{s}")) for s in c["cart"]])
        bml = np.array([code(r.get(f"V{v}MBMS{s}")) for s in c["bml"]])
        ost = np.array([code(r.get(f"V{v}MOS{s}")) for s in c["ost"]])
        if np.isfinite(cart).any():
            size, ft = np.floor(cart), np.round((cart - np.floor(cart)) * 10)
            out[f"{roi}|cart_size"] = np.nanmax(size) / 3
            out[f"{roi}|cart_full"] = np.nanmax(ft) / 3
            out[f"{roi}|cart_sum"] = min(np.nansum(size) / (3 * len(c["cart"])) * 2, 1.0)
        if np.isfinite(bml).any():
            out[f"{roi}|bml"] = np.nanmax(bml) / 3
        if np.isfinite(ost).any():
            out[f"{roi}|osteophyte"] = np.nanmax(ost) / 3
    for side, roi in (("M", "Medial Meniscus"), ("L", "Lateral Meniscus")):
        t = np.array([code(r.get(f"V{v}MMT{side}{p}")) for p in "ABP"])
        if np.isfinite(t).any():
            out[f"{roi}|tear"] = float(np.isin(t, [2, 3, 4, 5]).any() or code(r.get(f"V{v}MMRT{side}")) == 1)
            out[f"{roi}|maceration"] = float(np.isin(t, [6, 7, 8]).any())
            out[f"{roi}|signal"] = float((t >= 1).any())
        x = code(r.get(f"V{v}MMX{side}M" if side == "M" else f"V{v}MMX{side}L"))
        if np.isfinite(x):
            out[f"{roi}|extrusion"] = x / 3
    acl = code(r.get(f"V{v}MACLTR"))
    if np.isfinite(acl):
        out["ACL|tear"] = acl / 2
    pop = code(r.get(f"V{v}MPOPCYS"))
    if np.isfinite(pop):
        out["Baker's|cyst"] = float(pop == 1)
    return out


def build_aux(visit="00"):
    m = pd.read_csv(os.path.join(MOAKS_DIR, f"kMRI_SQ_MOAKS_BICL{visit}.txt"), sep="|", low_memory=False)
    m["side"] = m.SIDE.astype(str).str.extract(r"(Right|Left)", expand=False).str.upper()
    rows = [dict(StudyInstanceUID=f"OAI_{r['ID']}_{r['side']}", **aux_targets(r, visit)) for r in m.to_dict("records")]
    return pd.DataFrame(rows).groupby("StudyInstanceUID").mean().reset_index()


# --------------------------------------------- graded convention labels (v4, the ones in use)

def oa_graded(r, v, comp):
    """1 convention-positive (lesion >= 10% of a subregion with full-thickness loss); 0.5 large partial-thickness
    lesion or a small full-thickness one; 0.25 small partial-thickness lesion; 0 no cartilage lesion."""
    cart = np.array([code(r.get(f"V{v}MCM{s}")) for s in COMP[comp]["cart"]])
    if not np.isfinite(cart).any():
        return np.nan
    size, ft = np.floor(cart), np.round((cart - np.floor(cart)) * 10)
    if ((size >= 2) & (ft >= 1)).any():
        return 1.0
    if (size >= 2).any() or ((size >= 1) & (ft >= 1)).any():
        return 0.5
    return 0.25 if (size >= 1).any() else 0.0


def meniscus_graded(r, v, side):
    """1 tear / maceration / root tear; 0.25 intrasubstance signal (degeneration) only; 0 normal."""
    t = [code(r.get(f"V{v}MMT{side}{p}")) for p in "ABP"]
    root = code(r.get(f"V{v}MMRT{side}"))
    if not np.isfinite(t).any():
        return np.nan
    if any(x in (2, 3, 4, 5, 6, 7, 8) for x in t) or root == 1:
        return 1.0
    return 0.25 if any(x == 1 for x in t) else 0.0


def build_graded(visit="00"):
    m = pd.read_csv(os.path.join(MOAKS_DIR, f"kMRI_SQ_MOAKS_BICL{visit}.txt"), sep="|", low_memory=False)
    m["side"] = m.SIDE.astype(str).str.extract(r"(Right|Left)", expand=False).str.upper()
    rows = []
    for r in m.to_dict("records"):
        acl, pop = code(r.get(f"V{visit}MACLTR")), code(r.get(f"V{visit}MPOPCYS"))
        rows.append(dict(StudyInstanceUID=f"OAI_{r['ID']}_{r['side']}",
                         **{"ACL": np.nan if not np.isfinite(acl) else {0: 0.0, 1: 0.5, 2: 1.0}[int(acl)],
                            "Medial Meniscus": meniscus_graded(r, visit, "M"), "Lateral Meniscus": meniscus_graded(r, visit, "L"),
                            "Medial OA": oa_graded(r, visit, "Medial OA"), "Lateral OA": oa_graded(r, visit, "Lateral OA"),
                            "PF OA": oa_graded(r, visit, "PF OA"),
                            "Effusion": graded(r.get(f"V{visit}MEFFWK")), "Synovitis": graded(r.get(f"V{visit}MSYIC")),
                            "Baker's": np.nan if not np.isfinite(pop) else float(pop == 1)}))
    d = pd.DataFrame(rows)
    lab = [c for c in d.columns if c != "StudyInstanceUID"]
    out = d.groupby("StudyInstanceUID")[lab].mean().reset_index()        # several reading projects: mean
    for c in ("MCL", "Contusion", "Fracture"):
        out[c] = np.nan
    return out


if __name__ == "__main__":
    main()
