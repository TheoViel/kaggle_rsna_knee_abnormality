"""Score a pipeline submission on the 58 / holdout: python score_sub.py sub.csv"""
import sys
import pandas as pd
from evaluate import Scorer
S = Scorer()
sub = pd.read_csv(sys.argv[1], index_col=0)
cols = [c for c in sub if sub[c].nunique() > 1]
res, m58, mh = S.auc(sub[cols])
print(f"{sys.argv[1]}: 58 {m58:.4f} hold {mh:.4f} ({len(cols)} findings)")
print("  " + "  ".join(f"{c[:8]} {a:.3f}/{b:.3f}" for c, (a, b) in res.items()))
