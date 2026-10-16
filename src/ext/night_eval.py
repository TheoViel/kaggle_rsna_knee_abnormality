"""Overnight evaluation (2026-10-14 night): seed-group recipes, synovitis variants, fracture z coverage; run from src/.

    python -m ext.night_eval

Every score uses validation runs (AUC on the 58 radiologist studies / the holdout, Claude labels >= 0.5). A recipe is
a list of votes; a vote is a model or a seed group (list), whose members' ranks are averaged first. Run ids of the
scheduler jobs are read from logs/sched/jobs/<name>.log (last run dir printed). The 4 findings the notebook blends
with b1_c1 also get "+ b1": 50/50 rank with b1_c1 (synovitis: with rank(family, b1_c1)).
"""

import os
import re

import pandas as pd
from sklearn.metrics import roc_auc_score

from crops.train import finding_of
from knee_cls.data import load_tables, make_splits

R = "../logs/crops_run/"
J = "../logs/sched/jobs/"


def job(name):
    p = J + name + ".log"
    if not os.path.exists(p):
        return None
    runs = re.findall(r"crops_run/([0-9-]+/[0-9]+)", open(p).read())
    return runs[-1] if runs and os.path.exists(R + runs[-1] + "/val_study_preds.csv") else None


def score(ref, c):
    run, _, planes = ref.partition("@")
    if planes:
        v = pd.read_csv(R + run + "/val_series_preds.csv")
        v = v[(v.cls == c) & v.plane.isin(planes.split(","))]
        return v.groupby("StudyInstanceUID").p.mean()
    t = pd.read_csv(R + run + "/val_study_preds.csv")
    t = t.set_index(t.columns[0])
    if c in t:
        return t[c]
    v = pd.read_csv(R + run + "/val_series_preds.csv")                      # per-ROI sub-targets -> max over ROIs
    v = v[v.cls.str.split("|").str[1].fillna("").map(finding_of) == c]
    return v.groupby(["StudyInstanceUID", "cls"]).p.mean().groupby("StudyInstanceUID").max()


def recipe(votes, c):
    out = 0
    for v in votes:
        members = [m for m in (v if isinstance(v, list) else [v]) if m]
        if not members:
            continue
        out = out + sum(score(m, c).rank(pct=True) for m in members).rank(pct=True)
    return out


def main():
    st, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(st, holdout_frac=0.1, seed=0)
    S = st.set_index("StudyInstanceUID")
    b1 = pd.read_csv("../input/b1_c1_sealed_58_predictions.csv").set_index("StudyInstanceUID")
    fam = pd.read_csv("../input/synovitis_expert58_components.csv").set_index("StudyInstanceUID").three_family_oof
    b1side = {c: b1[c].rank(pct=True) for c in ("Fracture", "Contusion", "Effusion")}
    b1side["Synovitis"] = (b1["Synovitis"].rank(pct=True) + fam.reindex(b1.index).rank(pct=True)) / 2

    o, x, d, e, t, f = "2026-10-09/", "2026-10-10/", "2026-10-11/", "2026-10-12/", "2026-10-13/", "2026-10-14/"
    s88 = [o + "75", x + "100", job("seed_88_val_s2")]
    mr85, pl86 = [d + "65", d + "73", d + "74"], [d + "91", d + "99", d + "100"]
    mk87, caf = [d + "30", d + "129", d + "130"], [x + "93", e + "6"]
    mk89, mk90 = [d + "33", d + "115", d + "117"], [d + "17", d + "116", d + "118"]
    b91 = [o + "84", job("seed_91_val_s1"), job("seed_91_val_s2")]
    sy, sc = [t + "44", t + "49"], [t + "77", t + "79"]
    s224 = [job("syn224_s0"), job("syn224_s1")]
    hoffa, nor3 = job("syn_hoffa_s0"), job("syn_nor3_s0")
    fb = [t + "94", t + "98"]
    f16, fu12 = [job("frz_s1n16_s0"), job("frz_s1n16_s1")], [job("frz_u12_s0"), job("frz_u12_s1")]
    lc, ax = [f + "70", f + "72", f + "73"], f + "251"

    tests = {
        "ACL": {"dataset_12: mr85 + 88": [mr85[0], s88[0]], "[mr85 x3] + [88 x3]": [mr85, s88]},
        "MCL": {"dataset_12: pl86 + 88": [pl86[0], s88[0]], "[pl86 x3] + [88 x3]": [pl86, s88]},
        "Medial Meniscus": {"dataset_12: mk87 + cafax87 + 88": [mk87[0], caf[0], s88[0]],
                            "[mk87 x3] + [cafax87 x2] + [88 x3]": [mk87, caf, s88]},
        "Lateral Meniscus": {"dataset_12: mk87 + cafax87@SC + 88": [mk87[0], caf[0] + "@Sagittal,Coronal", s88[0]],
                             "[mk87 x3] + [cafax87@SC x2] + [88 x3]": [mk87, [c + "@Sagittal,Coronal" for c in caf], s88]},
        "Medial OA": {"dataset_12: 88 + mk89 + mk90": [s88[0], mk89[0], mk90[0]],
                      "[88 x3] + [mk89 x3] + [mk90 x3]": [s88, mk89, mk90]},
        "Lateral OA": {"dataset_12: [lc x3]": [lc]},
        "PF OA": {"dataset_12: mk90 + mk89 + ax0": [mk90[0], mk89[0], ax], "[mk90 x3] + [mk89 x3] + ax0": [mk90, mk89, ax]},
        "Baker's": {"dataset_12: 88 + 91": [s88[0], b91[0]], "[88 x3] + [91 x3]": [s88, b91]},
        "Synovitis": {"dataset_12: [sy x2] + [sc x2] (each model a vote)": sy + sc,
                      "+ [224 px x2] as 1 vote": sy + sc + [s224],
                      "+ fat pad crop (hoffa) as 1 vote": sy + sc + [hoffa],
                      "+ 224 px + fat pad": sy + sc + [s224, hoffa],
                      "160 px sy seed 0 alone": [sy[0]], "224 px seed 0 alone": [s224[0]], "224 px [x2]": [s224],
                      "fat pad crop alone": [hoffa], "3-way noisy-OR labels, seed 0": [nor3]},
        "Fracture": {"dataset_12: [fb x2] (8 strided)": [fb], "16 strided [x2]": [f16], "12 uniform [x2]": [fu12],
                     "8 + 16 strided": [fb, f16], "8 strided + 12 uniform": [fb, fu12]},
    }
    rows = []
    for c, variants in tests.items():
        G = S.loc[gt_ids, f"gt_{c}"].dropna().round()
        H = (S.loc[hold_ids, f"target_{c}"].dropna() >= 0.5).astype(int)
        for name, votes in variants.items():
            flat = [m for v in votes for m in (v if isinstance(v, list) else [v])]
            if not any(flat):
                continue
            missing = sum(m is None for m in flat)
            p = recipe(votes, c)
            r = dict(finding=c, recipe=name + (f"  [{missing} run(s) missing]" if missing else ""),
                     gt=roc_auc_score(G, p.reindex(G.index)), hold=roc_auc_score(H, p.reindex(H.index)))
            if c in b1side:
                q = b1side[c].reindex(G.index).rank(pct=True)
                r["+ b1"] = roc_auc_score(G, (p.reindex(G.index).rank(pct=True) + q) / 2)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("../output/locations/night_eval.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
