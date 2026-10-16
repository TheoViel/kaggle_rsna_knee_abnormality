"""Test-time augmentation check on validation runs (single GPU); run from src/.

    python -m ext.tta_eval --runs 2026-10-13/44,2026-10-11/17 [--n 8] [--scale 0.5]

For each run: its validation crops (as main_crops builds them: the 58 + the holdout, its crop sets and planes),
predicted once without augmentation (= the run's own validation predictions) and --n times with the training
augmentations (crops.train.gpu_augment: rotation, scale, shift, gamma, contrast, brightness) shrunk by --scale.
Flips are not used: training never flips. Per-series probabilities are averaged over the passes, then per study as
the pipeline does (mean over series; per-ROI sub-targets: max over ROIs). Prints AUC on the 58 / holdout per finding
for: plain, TTA mean, TTA mean + plain.
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from crops.train import GROUPS, CropDataset, build_model, center_view, finding_of, gpu_augment, load_crops, slice_window
from knee_cls.data import load_tables, make_splits
from main_crops import DEFAULTS, training_targets

R = "../logs/crops_run/"


def val_dataset(a, studies):
    crops, info = load_crops(a.crops)
    info = info[info.Anatomical_Plane.isin(a.planes.split(","))]
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    classes = GROUPS[a.group]
    val_t = training_targets("claude", studies, classes)
    for c in classes:
        F = finding_of(c.split("|")[1]) if "|" in c else None
        if F is not None:
            val_t[c] = training_targets("claude", studies, [F])[F]
    return CropDataset(crops, info[info.StudyInstanceUID.isin(set(gt_ids) | set(hold_ids))], classes, val_t), classes


@torch.no_grad()
def run_passes(model, ds, a, n, scale, device, bs=128):
    aug = {k: v for k, v in a.augmentations.items()}
    aug = dict(rot_deg=aug["rot_deg"] * scale, scale_range=[1 - (1 - aug["scale_range"][0]) * scale, 1 + (aug["scale_range"][1] - 1) * scale],
               shift=aug["shift"] * scale, gamma=aug["gamma"] * scale, contrast=aug["contrast"] * scale, brightness=aug["brightness"] * scale)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8)
    plain, tta = [], []
    for b in loader:
        x = b["x"].to(device).float() / 255
        x = slice_window(x, getattr(a, "depth_out", 0), 0)
        roi, plane = b["roi"].to(device), b["plane"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            plain.append(torch.sigmoid(model(center_view(x, a.ctx or 1.0), roi, plane).float()).cpu())
            ps = [torch.sigmoid(model(gpu_augment(x, aug, a.ctx or 1.0, 0.0), roi, plane).float()).cpu() for _ in range(n)]
        tta.append(torch.stack(ps).mean(0))
    return torch.cat(plain).numpy(), torch.cat(tta).numpy()


def study_scores(items, p):
    df = items[["StudyInstanceUID", "cls"]].assign(p=p)
    s = df.groupby(["StudyInstanceUID", "cls"]).p.mean().reset_index()
    s["F"] = [finding_of(c.split("|")[1]) if "|" in c else c for c in s.cls]
    return s.groupby(["StudyInstanceUID", "F"]).p.max().unstack()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.5, help="augmentation strength vs training")
    ap.add_argument("--out", default="../output/locations/tta_eval.csv")
    args = ap.parse_args()
    device = torch.device("cuda")
    studies, _ = load_tables()
    _, gt_ids, hold_ids = make_splits(studies, holdout_frac=0.1, seed=0)
    S = studies.set_index("StudyInstanceUID")
    rows = []
    for run in args.runs.split(","):
        a = SimpleNamespace(**{**DEFAULTS, **json.load(open(R + run + "/args.json"))})
        ds, classes = val_dataset(a, studies)
        model = build_model(vars(a), len(classes), pretrained=False)
        model.load_state_dict(torch.load(R + run + "/ckpt_final.pt", map_location="cpu"))
        model = model.to(device).eval()
        plain, tta = run_passes(model, ds, a, args.n, args.scale, device)
        sp, st_ = study_scores(ds.items, plain), study_scores(ds.items, tta)
        for F in sp.columns:
            if f"gt_{F}" not in S:
                continue
            G = S.loc[gt_ids, f"gt_{F}"].dropna().round()
            H = (S.loc[hold_ids, f"target_{F}"].dropna() >= 0.5).astype(int)
            r = dict(run=run, group=a.group, finding=F)
            for k, p in (("plain", sp[F]), ("tta", st_[F]), ("tta+plain", (sp[F].rank(pct=True) + st_[F].rank(pct=True)) / 2)):
                r[k] = f"{roc_auc_score(G, p.reindex(G.index)):.3f} / {roc_auc_score(H, p.reindex(H.index)):.3f}"
                r[k + "_gt"], r[k + "_hold"] = roc_auc_score(G, p.reindex(G.index)), roc_auc_score(H, p.reindex(H.index))
            rows.append(r)
            print(f"{run} {F:18s} plain {r['plain']}  tta {r['tta']}  tta+plain {r['tta+plain']}", flush=True)
        del model
        torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    d.to_csv(args.out, index=False)
    for k in ("plain", "tta", "tta+plain"):
        print(f"mean over {len(d)} (run, finding): {k:10s} {d[k + '_gt'].mean():.4f} / {d[k + '_hold'].mean():.4f}")


if __name__ == "__main__":
    main()
