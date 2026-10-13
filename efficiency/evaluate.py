"""Offline scoring of fast-pipeline choices from the validation runs' per-series predictions, run from efficiency/.

    python evaluate.py

Each validation crop run wrote val_series_preds.csv (study, series, cls, plane, p) on the 58 radiologist studies and the
holdout. Restricting those rows to a series subset and rank-averaging a model subset gives the score of a pipeline that
reads only those series and runs only those models (pipeline parity with the training crops: Pearson >= 0.9995).

Efficiency = AUC / (0.5 - max AUC) + runtime / 32400, with max AUC = full ensemble's 58 score + 0.005 (user's surrogate):
0.001 AUC is worth about 65 s of runtime.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(os.path.join(ROOT, "src"))

from knee_cls.data import load_tables, make_splits  # noqa: E402
from params import CLASSES  # noqa: E402

RUNS = os.path.join(ROOT, "logs", "crops_run")
# Validation counterparts of the submitted fullfit models (dataset_2_moaks_mrnet_men_frac).
VAL = {"mr85": "2026-10-11/65", "pl86": "2026-10-11/91", "mk87": "2026-10-11/129", "cafax87": "2026-10-10/93",
       "88": "2026-10-09/75", "mk89": "2026-10-11/33", "mk90": "2026-10-11/17", "91": "2026-10-09/84",
       "fr9a": "2026-10-12/25", "fr9b": "2026-10-12/26",
       "j27": "2026-10-10/27", "j26": "2026-10-10/26",           # main agent's joint4 (sag+cor+ax / sag+cor)
       "jc1": "../bis/crops_run/2026-10-12/1", "jc2": "../bis/crops_run/2026-10-12/2"}   # joint4, Codex labels
SUBMITTED = {"ACL": ["mr85", "88"], "MCL": ["pl86", "88"], "Medial Meniscus": ["mk87", "cafax87", "88"],
             "Lateral Meniscus": ["mk87", "cafax87@Sagittal,Coronal", "88"], "Medial OA": ["88", "mk89", "mk90"],
             "Lateral OA": ["mk89"], "PF OA": ["mk90"], "Baker's": ["88", "91"], "Fracture": ["fr9a", "fr9b"]}
EXTRA = ["Effusion", "Synovitis", "Contusion", "Fracture"]

_cache = {}


def series_preds(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(os.path.join(RUNS, VAL[name], "val_series_preds.csv"))
    return _cache[name]


def study_scores(name, finding, keep=None, planes=None):
    """Study probability of `finding` from model `name`: series mean (per ROI for "<ROI>|finding" rows, then max)."""
    d = series_preds(name)
    if keep is not None:
        d = d[d.SeriesInstanceUID.isin(keep)]
    if planes:
        d = d[d.plane.isin(planes)]
    direct = d[d.cls == finding]
    if len(direct):
        return direct.groupby("StudyInstanceUID").p.mean()
    sub = d[d.cls.str.endswith("|" + finding.lower())]
    return sub.groupby(["StudyInstanceUID", "cls"]).p.mean().groupby(level=0).max()


def combine(recipe, keep=None, index=None):
    out = {}
    for c, models in recipe.items():
        cols = []
        for m in models:
            name, planes = (m.split("@") + [None])[:2]
            cols.append(study_scores(name, c, keep, planes.split(",") if planes else None).reindex(index))
        out[c] = cols[0] if len(cols) == 1 else sum(x.rank(pct=True) for x in cols) / len(cols)
    return pd.DataFrame(out, index=index)


class Scorer:
    def __init__(self):
        studies, _ = load_tables()
        _, gt, hold = make_splits(studies, 0.1, 0)
        st = studies.set_index("StudyInstanceUID")
        self.gt, self.hold = st.loc[gt], st.loc[hold]
        self.index = pd.Index(list(gt) + list(hold), name="StudyInstanceUID")
        self.b1c1 = pd.read_csv(os.path.join(ROOT, "input", "b1_c1_sealed_58_predictions.csv"), index_col=0)
        ser = pd.read_csv(os.path.join(ROOT, "output", "series_index.csv"))
        self.series = ser[ser.StudyInstanceUID.isin(self.index)].reset_index(drop=True)

    def auc(self, pred):
        """-> ({finding: (58, holdout)}, macro 58, macro holdout) over the findings in pred."""
        res = {}
        for c in pred:
            p = pred[c]
            g = p.reindex(self.gt.index)
            a58 = roc_auc_score(self.gt[f"gt_{c}"], g.fillna(g.median()))
            h = p.reindex(self.hold.index)
            ah = roc_auc_score(self.hold[f"target_{c}"] >= 0.5, h.fillna(h.median())) if h.notna().any() else np.nan
            res[c] = (a58, ah)
        return res, np.mean([v[0] for v in res.values()]), np.nanmean([v[1] for v in res.values()])

    def reference(self):
        """The submitted ensemble on the 58: recipe (validation counterparts) on 8 ROI findings, b1_c1 on effusion /
        synovitis / contusion, fracture = rank average of b1_c1 and fr9a + fr9b."""
        p = combine(SUBMITTED, index=self.index)
        g = self.gt.index
        for c in ["Effusion", "Synovitis", "Contusion"]:
            p.loc[g, c] = self.b1c1.loc[g, c].values
        f = (p.loc[g, "Fracture"].rank(pct=True) + self.b1c1.loc[g, "Fracture"].rank(pct=True)) / 2
        p.loc[g, "Fracture"] = f.values
        return p[CLASSES]

    def pick(self, rule):
        """One sagittal + one coronal series per study (axial never) -> set of SeriesInstanceUIDs."""
        s = self.series[self.series.Anatomical_Plane.isin(["Sagittal", "Coronal"])]
        if rule == "all_sc":
            return set(s.SeriesInstanceUID)
        if rule == "all":
            return set(self.series.SeriesInstanceUID)
        key = {"fs": -s.Fluid_Sensitive, "nofs": s.Fluid_Sensitive, "most": -s.n_slices, "least": s.n_slices,
               "first": pd.Series(0, index=s.index)}[rule]
        s = s.assign(k=key.values).sort_values(["StudyInstanceUID", "Anatomical_Plane", "k", "SeriesInstanceUID"])
        return set(s.groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(1).SeriesInstanceUID)


def report(name, res, m58, mh, ref=None):
    line = f"{name:44s} 58 {m58:.4f}  hold {mh:.4f}"
    if ref is not None:
        line += f"  (d58 {m58 - ref:+.4f})"
    print(line)


if __name__ == "__main__":
    S = Scorer()
    ref = S.reference()
    res, m58, mh = S.auc(ref)
    print("max AUC surrogate:", round(m58 + 0.005, 4))
    report("reference (submitted, 12 findings)", res, m58, mh)
    print({c: round(v[0], 3) for c, v in res.items()})
