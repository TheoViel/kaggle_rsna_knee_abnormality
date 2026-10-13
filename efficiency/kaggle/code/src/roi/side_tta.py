"""Infer the knee side at test time with the canonical U-Net itself, run from src/.

    python -m roi.side_tta --run ../logs/roi_run/2026-10-08/3/

The ROI U-Net is trained on volumes shown as right knees. Run on a series as
stored and mirrored along its left-right axis, it should be sharper (higher mean
probability over its boxes) on whichever version is a right knee -- the other is
anatomy it never saw. Side = R if the unmirrored version scores higher; per study,
the score differences of its series are summed (one knee per study).

Scored against the DICOM Laterality tag on the validation studies that have one.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from roi.data import flip_axis, load_volumes, split_series
from roi.laterality import knee_side
from roi.train import build_model


@torch.no_grad()
def sharpness(model, vols, device, threshold=0.5):
    """(N, 1, D, S, S) float volumes -> (N,) mean over classes of the mean probability above threshold."""
    with torch.autocast(device.type, dtype=torch.bfloat16):
        p = torch.sigmoid(model(vols).float())
    on = (p >= threshold).float()
    per_class = (p * on).sum((2, 3, 4)) / on.sum((2, 3, 4)).clamp_min(1)
    return per_class.mean(1)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--run", required=True)
    a.add_argument("--batch", type=int, default=16)
    args = a.parse_args()
    device = torch.device("cuda")
    sd = torch.load(os.path.join(args.run, "ckpt_final.pt"), map_location=device)
    model = build_model(num_classes=sd["decoder.seg_layers.4.weight"].shape[0]).to(device).eval()
    model.load_state_dict(sd)

    vols, index = load_volumes()
    val = split_series(index)[1]
    tags = knee_side()
    val = val[val.StudyInstanceUID.isin(tags.index)].reset_index(drop=True)

    diffs = []
    t0 = time.time()
    for i in range(0, len(val), args.batch):
        rows = val.iloc[i:i + args.batch]
        x = torch.from_numpy(np.stack([np.array(vols[r]) for r in rows.row])).float().div(255)[:, None]
        xf = x.clone()
        for j, sag in enumerate(rows.Anatomical_Plane.eq("Sagittal")):
            xf[j] = x[j].flip(flip_axis(sag) + 1)
        x, xf = x.to(device), xf.to(device)
        diffs.append((sharpness(model, x, device) - sharpness(model, xf, device)).cpu().numpy())
    dt = time.time() - t0
    val["diff"] = np.concatenate(diffs)
    val["truth"] = val.StudyInstanceUID.map(tags)
    series_acc = ((val["diff"] > 0) == (val.truth == "R")).mean()
    study = val.groupby("StudyInstanceUID").agg(diff=("diff", "sum"), truth=("truth", "first"))
    study_acc = ((study["diff"] > 0) == (study.truth == "R")).mean()
    print(f"series-level accuracy {series_acc:.4f} ({len(val)} series), "
          f"study-level {study_acc:.4f} ({len(study)} studies); "
          f"{1000 * dt / len(val):.1f} ms per series for both passes")


if __name__ == "__main__":
    main()
