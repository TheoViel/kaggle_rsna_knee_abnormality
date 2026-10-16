"""Scheduler lines (logs/sched) for out-of-fold pseudo-labels of a weight package's recipe, run from src/.

    python -m pl.kfold_jobs --weights ../output/dataset_8 --n_folds 4 [--seed_from 0] >> ../logs/sched/queue.txt

One k-fold run per distinct model setup of the recipe (one seed: the first model of each setup), for every fold.
Each command reproduces the package model's training arguments (its args.json, fullfit off) on 2 GPUs: the
per-GPU batch is doubled for models trained on 4 GPUs (effective batch unchanged). CAFormer capsule models do
not fit at 2 x 64 (OOM): 2 x 32 (effective batch 64 instead of 128).
"""

import argparse
import json
import os

SKIP = {"output_dir", "log_path", "z_sampling", "augmentations", "fullfit", "n_folds", "fold", "seed"}
TWO_GPU = {"sy0", "sy1", "ef1", "fb0", "fb1", "ax0", "lc0", "lc1", "lc2"}   # package models trained on 2 GPUs (batch already per 2 GPUs)


def normalised(a):
    """args.json -> every DEFAULTS key (older runs lack newer keys: their default), JSON-comparable."""
    from main_crops import DEFAULTS
    return json.loads(json.dumps({k: a.get(k, v) for k, v in DEFAULTS.items()}, default=str))


def setups(weights):
    """Distinct setups of the package: {first model name: args}; seeds of one setup collapse."""
    out, seen = {}, set()
    for m in sorted(os.listdir(os.path.join(weights, "crops"))):
        a = json.load(open(os.path.join(weights, "crops", m, "args.json")))
        key = json.dumps({k: v for k, v in normalised(a).items() if k not in SKIP | {"batch_size"}}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out[m] = a
    return out


def command(m, a, n_folds, fold, seed):
    from main_crops import DEFAULTS
    bs = a["batch_size"] if m in TWO_GPU else 2 * a["batch_size"]
    if "caformer" in a["backbone"] and "cap_" in a["crops"]:
        bs = 32
    a, d = normalised(a), normalised(DEFAULTS)
    args = {k: v for k, v in a.items() if k not in SKIP and d[k] != v and k != "batch_size"}
    if a["augmentations"] != d["augmentations"]:
        args["augmentations"] = "'" + json.dumps(a["augmentations"]) + "'"
    flags = " ".join(f"--{k} {v}" for k, v in args.items())
    return (f"../.venv/bin/torchrun --standalone --nproc_per_node=$NP --master_port $PORT main_crops.py "
            f"{flags} --batch_size {bs} --seed {seed} --n_folds {n_folds} --fold {fold}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="../output/dataset_8")
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated model names")
    a = ap.parse_args()
    S = setups(a.weights)
    order = sorted(S, key=lambda m: ("joint4" in S[m]["group"], m))      # the slow joint4 model last
    if a.only:
        order = [m for m in order if m in a.only.split(",")]
    for fold in range(a.n_folds):
        for m in order:
            print(f"pl_{m}_s{a.seed}_f{fold}\t{command(m, S[m], a.n_folds, fold, a.seed)}")


if __name__ == "__main__":
    main()
