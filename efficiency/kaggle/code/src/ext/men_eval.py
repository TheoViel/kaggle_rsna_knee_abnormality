"""Menisci recipe check: does an extra menisci model help on top of dataset_2_moaks_mrnet_men? Run from src/.

    python -m ext.men_eval --candidates 2026-10-10/93,2026-10-12/NN [--names caf_ax_s0,caf_ax_s1]

Recipe of dataset_2_moaks_mrnet_men for the menisci (validation counterparts): medial and lateral meniscus =
rank average of [MOAKS CoAtNet menisci, "all" model]. Each candidate run (a non-fullfit menisci run with
val_study_preds.csv) is added as a third model, for both menisci. To check that a gain is not tied to one
lucky seed of the base menisci model, the base is also evaluated with the other MOAKS CoAtNet seeds and two
plain CoAtNet seeds. Prints AUC on the 58 radiologist studies / the holdout (Claude labels >= 0.5), per base
seed, plus the mean over base seeds.

--lm_planes Sagittal,Coronal: the candidates' lateral-meniscus score is averaged over their sagittal and coronal
series only (from val_series_preds.csv), as a recipe entry "name@Sagittal,Coronal" does at inference.
"""

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables, make_splits

R = "../logs/crops_run/"
ALL = "2026-10-09/75"                                     # the "all" model (88's validation counterpart)
BASE_SEEDS = {"MOAKS s0 (packaged)": "2026-10-11/30", "MOAKS s1": "2026-10-11/129", "MOAKS s2": "2026-10-11/130",
              "plain s0": "2026-10-09/81", "plain s1": "2026-10-11/71"}
MEN = ["Medial Meniscus", "Lateral Meniscus"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, help="comma-separated run ids (YYYY-MM-DD/N)")
    ap.add_argument("--names", default="", help="comma-separated labels for the candidates")
    ap.add_argument("--lm_planes", default="", help="planes the candidates' lateral-meniscus score is averaged over")
    a = ap.parse_args()
    cands = a.candidates.split(",")
    names = a.names.split(",") if a.names else cands
    st, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(st, holdout_frac=0.1, seed=0)
    S = st.set_index("StudyInstanceUID")
    cache = {}

    def study(r, c):
        if a.lm_planes and c == "Lateral Meniscus" and r in cands:
            key = (r, c, a.lm_planes)
            if key not in cache:
                v = pd.read_csv(R + r + "/val_series_preds.csv")
                v = v[(v.cls == c) & v.plane.isin(a.lm_planes.split(","))]
                cache[key] = v.groupby("StudyInstanceUID").p.mean()
            return cache[key]
        if r not in cache:
            cache[r] = pd.read_csv(R + r + "/val_study_preds.csv").set_index("StudyInstanceUID")
        return cache[r][c]

    def score(runs, c):
        p = sum(study(r, c).rank(pct=True) for r in runs) / len(runs)
        g = S.loc[gt_ids, f"gt_{c}"].dropna()
        h = S.loc[hold_ids, f"target_{c}"].dropna()
        return roc_auc_score(g.round(), p.reindex(g.index)), roc_auc_score(h.round(), p.reindex(h.index))

    rows = []
    for bname, b in BASE_SEEDS.items():
        for label, extra in [("base", [])] + [(n, [r]) for n, r in zip(names, cands)]:
            row = {"base seed": bname, "variant": label}
            for c in MEN:
                row[f"{c} 58"], row[f"{c} hold"] = score([b, ALL] + extra, c)
            row["mean 58"] = (row["Medial Meniscus 58"] + row["Lateral Meniscus 58"]) / 2
            row["mean hold"] = (row["Medial Meniscus hold"] + row["Lateral Meniscus hold"]) / 2
            rows.append(row)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(d.round(4).to_string(index=False))
    print("\nMean over the 5 base seeds:")
    print(d.groupby("variant", sort=False)[[c for c in d.columns if c not in ("base seed", "variant")]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
