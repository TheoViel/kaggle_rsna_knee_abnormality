"""Series rules by bytes read vs AUC (recipe eff)."""
import pandas as pd, numpy as np
from evaluate import Scorer, combine
from additions import EFF
S = Scorer()
s = S.series[S.series.Anatomical_Plane.isin(["Sagittal", "Coronal"])].copy()
s["mb"] = np.minimum(s.n_slices, 64) * s.rows * s.cols * 2 / 1e6
def pick(key):
    d = s.assign(k=key).sort_values(["StudyInstanceUID", "Anatomical_Plane", "k", "SeriesInstanceUID"])
    return d.groupby(["StudyInstanceUID", "Anatomical_Plane"]).head(1)
rules = {"fs": -s.Fluid_Sensitive, "nofs": s.Fluid_Sensitive, "least bytes": s.mb, "fs, then least bytes": -s.Fluid_Sensitive * 1e6 + s.mb,
         "most bytes": -s.mb}
for n, k in rules.items():
    d = pick(k.values)
    res, a58, ah = S.auc(combine(EFF, set(d.SeriesInstanceUID), S.index))
    print(f"{n:22s} {d.mb.sum() / S.index.size:5.1f} MB/study  58 {a58:.4f} hold {ah:.4f}")
