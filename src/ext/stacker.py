"""Cross-finding stacker over the out-of-fold pseudo-labels (pl.assemble), run from src/.

    python -m ext.stacker [--pl ../output/pl_dataset_8/pl_mean.csv]

Per finding, a small model maps the 12 out-of-fold finding scores of a study to that finding. It is trained on the
training studies other than the 58, against their soft training-convention labels (Claude), and scored on the 58
radiologist studies, whose 12 scores are out-of-fold too. Variants:
  base        the finding's own out-of-fold score (no stacking)
  lr          logistic regression on the logits of the 12 scores (L2, C = 1)
  lr_own+k    logistic regression on the own score + only the k findings most correlated with it in training
  gbm         histogram gradient boosting on the 12 scores (shallow: depth 3, 200 iterations, lr 0.05)
  blend       rank mean of base and lr: keeps the own model's ranking, adds the cross-finding signal
On the 58 it reports AUC per finding, and the cross-talk of the OA compartments (Spearman between their scores; the
radiologists: medial-lateral 0.32, PF-medial 0.37). A 5-fold CV on the non-58 studies against the Claude labels is
printed for reference (the report convention, which the stacker is trained on).
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from knee_cls.data import load_tables

CLASSES = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_predict(kind, Xtr, ytr, Xte, own, k_cols=None):
    if kind == "base":
        return Xte[own]
    cols = list(Xtr.columns) if k_cols is None else k_cols
    if kind.startswith("lr"):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(logit(Xtr[cols]), ytr)
        return pd.Series(m.predict_proba(logit(Xte[cols]))[:, 1], index=Xte.index)
    if kind == "gbm":
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(Xtr[cols], ytr)
        return pd.Series(m.predict_proba(Xte[cols])[:, 1], index=Xte.index)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pl", default="../output/pl_dataset_8/pl_mean.csv")
    ap.add_argument("--out", default="../output/locations/stacker_eval.csv")
    a = ap.parse_args()
    X = pd.read_csv(a.pl).set_index("StudyInstanceUID")[CLASSES]
    st, _ = load_tables()
    S = st.set_index("StudyInstanceUID")
    gt_ids = S.index[S.is_gt]
    tr_ids = X.index.difference(gt_ids)
    rows, preds = [], {}
    for c in CLASSES:
        y = (S.loc[tr_ids, f"target_{c}"] >= 0.5).astype(int)
        ok = y.index[S.loc[y.index, f"target_{c}"].notna() & X.loc[y.index].notna().all(axis=1)]
        Xtr, ytr = X.loc[ok], y.loc[ok]
        G = S.loc[gt_ids, f"gt_{c}"].dropna().round()
        Xte = X.loc[G.index]
        corr = Xtr.corrwith(Xtr[c], method="spearman").drop(c).abs().sort_values(ascending=False)
        variants = {"base": None, "lr": None, "gbm": None}
        for k in (1, 2, 3):
            variants[f"lr_own+{k}"] = [c] + list(corr.index[:k])
        r = {"finding": c, "partners (top 3)": ", ".join(f"{i} {corr[i]:.2f}" for i in corr.index[:3])}
        for kind, cols in variants.items():
            p = fit_predict(kind.split("_")[0] if kind.startswith("lr_") else kind, Xtr, ytr, Xte, c, cols)
            preds[(c, kind)] = p
            r[kind] = roc_auc_score(G, p)
            # reference: 5-fold CV on the non-58 studies against the Claude labels
            if kind in ("base", "lr"):
                cv = []
                for tr, te in KFold(5, shuffle=True, random_state=0).split(Xtr):
                    pp = fit_predict(kind, Xtr.iloc[tr], ytr.iloc[tr], Xtr.iloc[te], c)
                    cv.append(roc_auc_score(ytr.iloc[te], pp))
                r[f"{kind} cv(claude)"] = np.mean(cv)
        bl = (preds[(c, "base")].rank(pct=True) + preds[(c, "lr")].rank(pct=True)) / 2
        preds[(c, "blend")] = bl
        r["blend"] = roc_auc_score(G, bl)
        rows.append(r)
    d = pd.DataFrame(rows).set_index("finding")
    d.to_csv(a.out)
    pd.set_option("display.width", 250)
    cols = ["base", "lr", "blend", "lr_own+1", "lr_own+2", "lr_own+3", "gbm", "base cv(claude)", "lr cv(claude)"]
    print(d[cols].round(3).to_string())
    print("\nmacro (12):", d[cols].mean().round(4).to_dict())
    print("partners:\n" + d["partners (top 3)"].to_string())
    G = S.loc[gt_ids]
    print("\nOA cross-talk on the 58 (Spearman): medial-lateral / PF-medial / PF-lateral")
    rad = (G["gt_Medial OA"].corr(G["gt_Lateral OA"], method="spearman"), G["gt_PF OA"].corr(G["gt_Medial OA"], method="spearman"),
           G["gt_PF OA"].corr(G["gt_Lateral OA"], method="spearman"))
    print(f"  radiologists {rad[0]:.2f} / {rad[1]:.2f} / {rad[2]:.2f}")
    for kind in ("base", "lr", "blend", "gbm"):
        m, l, p = (preds[(f, kind)] for f in ("Medial OA", "Lateral OA", "PF OA"))
        idx = m.index.intersection(l.index).intersection(p.index)
        print(f"  {kind:6s}       {m[idx].corr(l[idx], method='spearman'):.2f} / {p[idx].corr(m[idx], method='spearman'):.2f} / "
              f"{p[idx].corr(l[idx], method='spearman'):.2f}")


if __name__ == "__main__":
    main()
