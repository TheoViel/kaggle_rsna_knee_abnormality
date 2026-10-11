"""Recipe-level validation scores (58 / holdout) of dataset_2_moaks-style recipes with swapped models, run from src/.

    python -m ext.recipe_eval

Each finding = mean of percentile ranks of its models' validation study predictions (non-fullfit runs),
as at inference. BASE is dataset_2_moaks with its non-fullfit counterparts; variants swap runs in.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

from knee_cls.data import load_tables, make_splits

R = "../logs/crops_run/"
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Baker's"]


def recipe(acl="2026-10-09/79", mcl="2026-10-09/80", men="2026-10-09/81", all_="2026-10-09/75",
           oa224="2026-10-11/33", oaax="2026-10-11/17", bakers="2026-10-09/84"):
    a = all_ if isinstance(all_, list) else [all_]
    return {"ACL": [acl] + a, "MCL": [mcl] + a, "Medial Meniscus": [men] + a, "Lateral Meniscus": [men] + a,
            "Medial OA": a + [oa224, oaax], "Lateral OA": [oa224], "PF OA": [oaax], "Baker's": [bakers] + a}


def score(rec, st, gt_ids, hold_ids, cache={}):
    S = st.set_index("StudyInstanceUID")
    out = {}
    for c in FINDINGS:
        ps = []
        for run in rec[c]:
            if run not in cache:
                cache[run] = pd.read_csv(R + run + "/val_study_preds.csv").set_index("StudyInstanceUID")
            ps.append(cache[run][c].rank(pct=True))
        r = sum(ps) / len(ps)
        g = S.loc[gt_ids, f"gt_{c}"].dropna()
        h = S.loc[hold_ids, f"target_{c}"].dropna()
        out[c] = (roc_auc_score(g.round(), r.reindex(g.index)), roc_auc_score(h.round(), r.reindex(h.index)))
    t = pd.DataFrame(out, index=["gt", "hold"]).T
    t.loc["macro"] = t.mean()
    return t


def main():
    st, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(st, holdout_frac=0.1, seed=0)
    d = "2026-10-11/"
    variants = {
        "base (dataset_2_moaks)": recipe(),
        "acl mrnet w2 s0": recipe(acl=d + "65"), "acl mrnet w2 s1": recipe(acl=d + "73"), "acl mrnet w2 s2": recipe(acl=d + "74"),
        "acl mrnet w4 s0": recipe(acl=d + "82"), "acl mrnet w4 s1": recipe(acl=d + "83"),
        "all oai4 s0": recipe(all_=d + "78"), "all oai4 s1": recipe(all_=d + "79"),
        "all mrnet w2": recipe(all_=d + "80"),
        "all oai4+mrnet": recipe(all_=d + "81"),
        "men skm w4": recipe(men=d + "88"),
        "men mrnet w2": recipe(men=d + "86"),
        "men mensoft s0": recipe(men=d + "84"), "men mensoft s1": recipe(men=d + "85"),
        "acl w2 + all oai4+mrnet + men skm w4": recipe(acl=d + "65", all_=d + "81", men=d + "88"),
        "acl w2 + all oai4+mrnet": recipe(acl=d + "65", all_=d + "81"),
    }
    rows = {}
    for k, rec in variants.items():
        t = score(rec, st, gt_ids, hold_ids)
        rows[k] = {**{c: f"{t.loc[c, 'gt']:.3f}" for c in FINDINGS}, "macro 58": round(t.loc["macro", "gt"], 4),
                   "macro hold": round(t.loc["macro", "hold"], 4)}
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
