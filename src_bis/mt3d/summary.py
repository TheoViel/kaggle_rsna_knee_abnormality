"""One row per mt3d run: what it changed and its scores, run from src_bis/.

    python -m mt3d.summary [--since 2026-10-10/7]

Columns: the arguments that differ from main_mt3d.DEFAULTS; the mean AUC over the
8 ROI findings on the 58 (`gt8`) and the holdout (`hold8`); the holdout mean
without lateral OA (`hold7`: its holdout labels (Claude) carry the medial / lateral
cross-talk, so they reward it); the 4 extra findings when trained (`gt4x`, `hold4x`);
and per ROI finding the 58 / holdout AUCs. Selection is on the holdout columns.
"""

import argparse
import glob
import json
import os

import pandas as pd

from main_mt3d import DEFAULTS
from roi.extract import ROI_CLASSES

MT = "../logs/bis/mt3d/"
ABBR = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA",
        "Lateral OA": "LOA", "PF OA": "PFOA", "Baker's": "Bak"}
SKIP = {"output_dir", "log_path", "augmentations", "num_workers", "log_every"}


def runs(since=None):
    out = []
    for d in sorted(glob.glob(MT + "*/*/"), key=lambda p: (p.split("/")[-3], int(p.split("/")[-2]))):
        name = "/".join(d.rstrip("/").split("/")[-2:])
        if since and (name.split("/")[0], int(name.split("/")[1])) < (since.split("/")[0], int(since.split("/")[1])):
            continue
        if not os.path.exists(d + "val_table.csv"):
            continue
        a = json.load(open(d + "args.json"))
        t = pd.read_csv(d + "val_table.csv", index_col=0)
        h = pd.read_csv(d + "history.csv")
        diff = {k: v for k, v in a.items() if k not in SKIP and k in DEFAULTS and str(v) != str(DEFAULTS[k])}
        roi = t.loc[[c for c in ROI_CLASSES if c in t.index]]
        x = t.loc[[c for c in t.index if c not in ROI_CLASSES]]
        r = dict(run=name, epochs=len(h), changes=" ".join(f"{k}={v}" for k, v in diff.items()),
                 gt8=roi["gt"].mean(), hold8=roi["hold"].mean(), hold7=roi.drop("Lateral OA")["hold"].mean(),
                 gt4x=x["gt"].mean() if len(x) else None, hold4x=x["hold"].mean() if len(x) else None)
        r.update({ABBR[c]: f"{roi["gt"][c]:.3f}/{roi["hold"][c]:.3f}" for c in roi.index})
        out.append(r)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default=None)
    a = p.parse_args()
    pd.set_option("display.width", 400)
    pd.set_option("display.max_colwidth", 80)
    print(runs(a.since).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
