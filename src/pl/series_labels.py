"""Series-level target factors from out-of-fold series predictions, run from src/.

    python -m pl.series_labels --weights ../output/dataset_13                       # -> output/series_pl/<model>.csv
    python -m pl.series_labels --variant v1 --alpha 0.25 [--teacher ensemble] [--gate 0.5]
                                                                                    # -> output/series_pl/v1/<model>.csv

Our crop models train every series of a study (and its ROI crops) on the study's label, whether or not the finding
shows on that series. The teacher's out-of-fold probability per (series, class) comes from clean k-fold runs
(pl.assemble.kfold_runs: the 58 and the holdout never trained on), seeds averaged:
  --teacher self      the setup's own k-fold runs (default; the first version)
  --teacher ensemble  per (series, class), the mean over every setup of the package that predicts that class and
                      has the series (rank-free probability mean; plane-restricted setups count where they apply)
Per study and class:
    factor(series) = 1 - alpha x (1 - p(series) / max over the study's series p)   (--within plane: its plane's series)
so the study's clearest series keeps 1 and a series showing nothing drops to 1 - alpha (alpha 0.5: the first
version). --gate g: studies whose max p is below g keep factor 1 on every series (the teacher does not see the finding
anywhere, so its series ratios are noise). main_crops --series_labels applies target x factor to the training items
whose study target is >= 0.5 (positives); negative studies keep their label on every series. Writes
output/series_pl/[<variant>/]<model>.csv (SeriesInstanceUID, cls, factor) for the first model of each setup
(pl.kfold_jobs.setups: seeds share it) and prints the coverage.
"""
import argparse
import os

import pandas as pd

from params import OUTPUT_FOLDER
from pl.assemble import key, kfold_runs
from pl.kfold_jobs import setups


def series_oof(runs):
    """(SeriesInstanceUID, cls) -> out-of-fold p, seeds averaged (a seed counts once all its folds are in)."""
    per_seed = {}
    for d, fold, seed, _ in runs:
        v = pd.read_csv(os.path.join(d, "val_series_preds.csv"))
        per_seed.setdefault(seed, []).append(v)
    n = max(len(v) for v in per_seed.values())
    full = [pd.concat(v) for v in per_seed.values() if len(v) == n]
    assert full, "no seed with all its folds"
    v = pd.concat(full)
    return v.groupby(["StudyInstanceUID", "SeriesInstanceUID", "plane", "cls"]).p.mean().reset_index(), len(full), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="../output/dataset_13")
    ap.add_argument("--variant", default="", help="output sub-folder ('' = output/series_pl/ itself)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--teacher", default="self", choices=["self", "ensemble"])
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--within", default="study", choices=["study", "plane"],
                    help="max p over the study's series, or over its series of the same plane (each plane's best = 1)")
    a = ap.parse_args()
    out = os.path.join(OUTPUT_FOLDER, "series_pl", a.variant)
    os.makedirs(out, exist_ok=True)
    runs = kfold_runs()
    S = setups(a.weights)
    oof = {}
    for m, args in S.items():
        r = runs.get(key(args))
        if r:
            oof[m], _, _ = series_oof(r)
        else:
            print(f"{m}: no clean k-fold runs, skipped")
    if a.teacher == "ensemble":
        ens = pd.concat(oof.values()).groupby(["StudyInstanceUID", "SeriesInstanceUID", "plane", "cls"]).p.mean()
    for m, v in oof.items():
        if a.teacher == "ensemble":           # the setup's own (series, class) rows, ensemble probabilities
            v = v.drop(columns="p").join(ens, on=["StudyInstanceUID", "SeriesInstanceUID", "plane", "cls"])
        grp = ["StudyInstanceUID", "cls"] + (["plane"] if a.within == "plane" else [])
        mx = v.groupby(grp).p.transform("max").clip(lower=1e-6)
        v["factor"] = 1 - a.alpha * (1 - v.p / mx)
        if a.gate:
            v.loc[mx < a.gate, "factor"] = 1.0
        v[["SeriesInstanceUID", "cls", "factor"]].to_csv(os.path.join(out, f"{m}.csv"), index=False)
        print(f"{m:10s} {S[m]['group']:15s} {v.SeriesInstanceUID.nunique()} series, mean factor {v.factor.mean():.3f}, "
              f"share at 1: {(v.factor >= 0.999).mean():.2f}")


if __name__ == "__main__":
    main()
