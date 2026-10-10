"""Study-level predictions of trained crop models on every study, run from src/.

    python -m crops.predict --runs 6,7,10 --gpus 0,1,2,3

For each run: the model is rebuilt from its args.json, its checkpoint loaded, and
every series of its planes (all studies, training and validation) predicted for
the findings of its group; a study's probability is the mean over its series.
Writes <run dir>/all_study_preds.csv (StudyInstanceUID x findings).

Note: on training studies these predictions are in-sample -- the model was
trained on them.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

RUNS_DIR = "../logs/crops_run/2026-10-09/"


@torch.no_grad()
def predict_run(run, device):
    from crops.train import GROUPS, CropDataset, build_model, load_crops

    d = os.path.join(RUNS_DIR, run)
    a = json.load(open(os.path.join(d, "args.json")))
    classes = GROUPS[a["group"]]
    crops, info = load_crops(a.get("crops", "r3"))
    info = info[info.Anatomical_Plane.isin(a["planes"].split(","))]
    dummy = {c: pd.Series(0.0, index=info.StudyInstanceUID.unique()) for c in classes}
    ds = CropDataset(crops, info, classes, dummy)
    sd = torch.load(os.path.join(d, "ckpt_final.pt"), map_location="cpu")
    model = build_model(a, len(classes))
    # Runs from before axial crops existed have a 2-plane embedding (sagittal, coronal).
    n_planes = sd["plane_embed.weight"].shape[0]
    if n_planes != model.plane_embed.num_embeddings:
        model.plane_embed = nn.Embedding(n_planes, model.plane_embed.embedding_dim)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    ps = []
    for b in dl:
        x = b["x"].to(device, non_blocking=True).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ps.append(torch.sigmoid(model(x, b["roi"].to(device), b["plane"].to(device)).float()).cpu().numpy())
    it = ds.items.assign(p=np.concatenate(ps))
    tab = it.groupby(["StudyInstanceUID", "cls"]).p.mean().unstack()
    tab.to_csv(os.path.join(d, "all_study_preds.csv"))
    print(f"run {run}: {tab.shape[0]} studies x {list(tab.columns)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--worker", type=int, default=None)
    a = p.parse_args()
    runs = a.runs.split(",")
    gpus = a.gpus.split(",")
    if a.worker is None and len(gpus) > 1:
        procs = [subprocess.Popen([sys.executable, "-m", "crops.predict", "--runs", ",".join(runs[i::len(gpus)]),
                                   "--gpus", g, "--worker", str(i)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=g))
                 for i, g in enumerate(gpus) if runs[i::len(gpus)]]
        sys.exit(max(p.wait() for p in procs))
    if len(gpus) == 1 and a.worker is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpus[0])
    for r in runs:
        predict_run(r, torch.device("cuda"))


if __name__ == "__main__":
    main()
