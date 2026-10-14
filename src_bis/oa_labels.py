"""OA targets from the per-compartment report extraction (output/bis/oa_reports/chunks/oa_out_*.json), run from src/.

    PYTHONPATH=. python ../src_bis/oa_labels.py [--write]

The extraction (spec: output/bis/oa_reports/oa_extraction_spec.md, Claude agents reading every report) gives per
compartment: cartilage grade, osteophytes, subchondral changes, whether OA is stated, an overall severity, plus an
"unlocalised OA" flag. Studies outside the extraction (Claude OA labels all < 0.05) have no OA mentioned: 0.

Variants (per compartment score in [0, 1]):
  sev      severity: none 0, mild 0.35, moderate 0.75, severe 1, unk 0.6
  sev_oa   max(sev, oa stated: yes 0.8 / possible 0.4), osteophytes large 0.8 / present 0.6 / small 0.4
  strict   moderate+ cartilage or osteophytes (any) or OA stated yes -> 1; mild / possible -> 0.3
  *_unloc  as above, and compartments with nothing stated in a study with unlocalised OA get the Claude label
           instead of 0 (the report says OA somewhere; the image decides where)
  *_mask   ... get NaN instead (no loss for that compartment)
  *_mix    mean of the variant and the Claude label
Prints, per variant: AUC of the report score vs the radiologist labels on the 58 (the label quality), its
medial-lateral correlation over all studies, and the positive rate (>= 0.5). --write saves each variant as
output/bis/pl_oa_<variant>.csv (pl_round2 with the 3 OA columns replaced; --labels pl:../bis/pl_oa_<variant>).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from params import INPUT_FOLDER, OUTPUT_FOLDER

OA = {"medial": "Medial OA", "lateral": "Lateral OA", "patellofemoral": "PF OA"}
SEV = {"none": 0.0, "mild": 0.35, "moderate": 0.75, "severe": 1.0, "unk": 0.6}


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "bis", "oa_reports", "chunks", "oa_out_*.json"))):
        for e in json.load(open(f)):
            r = {"StudyInstanceUID": e["StudyInstanceUID"], "unloc": e.get("unlocalised_oa", "no")}
            for k, c in OA.items():
                comp = e.get(k) or {}
                for f_ in ("cartilage", "osteophytes", "subchondral", "oa", "severity"):
                    r[f"{c}|{f_}"] = comp.get(f_)
            rows.append(r)
    return pd.DataFrame(rows).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")


def nothing(d, c):
    """Compartment with nothing stated."""
    return (d[f"{c}|severity"].isin(["none", None]) & d[f"{c}|cartilage"].isin(["not_mentioned", None]) &
            d[f"{c}|osteophytes"].isin(["not_mentioned", None]) & d[f"{c}|oa"].isin(["not_mentioned", None]))


def score(d, c, kind):
    sev = d[f"{c}|severity"].map(SEV).fillna(0.0)
    if kind == "sev":
        return sev
    ost = d[f"{c}|osteophytes"].map({"large": 0.8, "present": 0.6, "small": 0.4}).fillna(0.0)
    oa = d[f"{c}|oa"].map({"yes": 0.8, "possible": 0.4}).fillna(0.0)
    if kind == "sev_oa":
        return pd.Series(np.maximum.reduce([sev.values, ost.values, oa.values]), index=d.index)
    if kind == "strict":
        hi = d[f"{c}|cartilage"].isin(["moderate", "severe"]) | d[f"{c}|osteophytes"].isin(["small", "large", "present"]) \
            | d[f"{c}|oa"].eq("yes") | d[f"{c}|severity"].isin(["moderate", "severe"])
        lo = d[f"{c}|cartilage"].isin(["mild", "present"]) | d[f"{c}|oa"].eq("possible") | d[f"{c}|severity"].isin(["mild", "unk"])
        return pd.Series(np.where(hi, 1.0, np.where(lo, 0.3, 0.0)), index=d.index)
    raise ValueError(kind)


def variants(d, claude, all_ids):
    out = {}
    for base in ("sev", "sev_oa", "strict"):
        for mode in ("", "_unloc", "_mask", "_mix"):
            t = pd.DataFrame(index=pd.Index(all_ids, name="StudyInstanceUID"))
            for c in OA.values():
                s = score(d, c, base).reindex(all_ids).fillna(0.0)
                if mode in ("_unloc", "_mask"):
                    unl = d.unloc.isin(["yes", "possible"]) & nothing(d, c)
                    unl = unl.reindex(all_ids).fillna(False).astype(bool)
                    s = s.where(~unl, claude[c].reindex(all_ids) if mode == "_unloc" else np.nan)
                if mode == "_mix":
                    s = (s + claude[c].reindex(all_ids)) / 2
                t[c] = s
            out[base + mode] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    d = load()
    train = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv")).set_index("StudyInstanceUID")
    claude = pd.read_csv(os.path.join(INPUT_FOLDER, "full_claude_context_v2_labels.csv")).set_index("StudyInstanceUID")
    all_ids = list(train.index)
    gt = train[list(OA.values())].dropna()
    print(f"extraction: {len(d)} studies; 58 covered: {gt.index.isin(d.index).sum()}")
    rows = []
    vs = {"claude": claude[list(OA.values())].reindex(all_ids), **variants(d, claude, all_ids)}
    for n, t in vs.items():
        r = {"variant": n}
        for c in OA.values():
            s = t.loc[gt.index, c]
            m = s.notna()
            r[c] = roc_auc_score(gt.loc[m[m].index, c], s[m])
            r[c + " pos"] = (t[c] >= 0.5).mean()
        r["MOA-LOA corr"] = t["Medial OA"].corr(t["Lateral OA"])
        rows.append(r)
        if a.write and n != "claude":
            pl = pd.read_csv(os.path.join(OUTPUT_FOLDER, "crops", "pl_round2.csv")).set_index("StudyInstanceUID")
            for c in OA.values():
                pl[c] = t[c].reindex(pl.index)
            pl.reset_index().to_csv(os.path.join(OUTPUT_FOLDER, "bis", f"pl_oa_{n}.csv"), index=False)
    print(pd.DataFrame(rows).set_index("variant").round(3).to_string())


if __name__ == "__main__":
    main()
