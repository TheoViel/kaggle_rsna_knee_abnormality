"""Collect crop-classifier runs: python -m crops.summary [--since 2026-10-09/2] -> one row per run, per-class gt / hold."""

import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="../logs/crops_run/")
    p.add_argument("--keys", default="group,backbone,img_size,head,planes,epochs,crops,lr,lr_backbone")
    a = p.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "*", "*", "")), key=lambda x: (x.split(os.sep)[-3], int(x.split(os.sep)[-2]))):
        t = os.path.join(d, "val_table.csv")
        if not os.path.exists(t) or not os.path.exists(os.path.join(d, "ckpt_final.pt")):
            continue
        args = json.load(open(os.path.join(d, "args.json")))
        tab = pd.read_csv(t, index_col=0)
        row = {"run": "/".join(d.split(os.sep)[-3:-1]), **{k: args.get(k) for k in a.keys.split(",")},
               "gt": tab["gt"].mean(), "hold": tab["hold"].mean()}
        for c, r in tab.iterrows():
            row[f"{c[:12]} gt"] = r["gt"]
            row[f"{c[:12]} hold"] = r["hold"]
        rows.append(row)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
