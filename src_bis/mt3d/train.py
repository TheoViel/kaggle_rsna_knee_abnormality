"""Data, losses and fit loop for the multitask 3D model (`mt3d.model.MT3D`).

One sample = one sagittal / coronal series: its joint crop (`mt3d.prepare`), its 8
ROI boxes (segmentation targets) and the study's soft labels for the 8 ROI
findings. Study scores are the mean over its series, as for the crop models,
scored by `crops.train.study_auc` (58 radiologist studies `gt`, holdout `hold`).
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from crops.train import study_auc
from mt3d.model import PLANES, box_masks
from mt3d.prepare import MT_DIR
from roi.data import gpu_augment
from roi.extract import ROI_CLASSES
from training.distributed import all_gather_lists, all_reduce_mean, unwrap_model


def load_crops(name):
    arr = np.load(os.path.join(MT_DIR, f"{name}.npy"), mmap_mode="r")
    boxes = np.load(os.path.join(MT_DIR, f"{name}_boxes.npy"))
    info = pd.read_csv(os.path.join(MT_DIR, f"{name}.csv"))
    return arr, boxes, info


class MTDataset(Dataset):
    def __init__(self, arr, boxes, info, targets, classes=ROI_CLASSES):
        self.arr, self.boxes = arr, boxes
        self.info = info.reset_index(drop=True)
        y = np.stack([self.info.StudyInstanceUID.map(targets[c]).to_numpy(float) for c in classes], 1)
        self.y = y.astype(np.float32)                       # NaN: no label (masked out)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, i):
        r = self.info.iloc[i]
        return {"x": torch.from_numpy(np.array(self.arr[r.row])), "boxes": torch.from_numpy(self.boxes[r.row]),
                "plane": torch.tensor(PLANES.index(r.Anatomical_Plane)),
                "meta": torch.tensor([r.Fluid_Sensitive, r.Fat_Suppression], dtype=torch.float32),
                "y": torch.from_numpy(self.y[i]), "idx": torch.tensor(i)}


def seg_loss(logits, target):
    """BCE + soft Dice, target (B, K, D, S, S) downsampled to the logits' grid (soft, area)."""
    t = F.adaptive_avg_pool3d(target, logits.shape[2:])
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits)
    inter = (p * t).sum((2, 3, 4))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3, 4)) + t.sum((2, 3, 4)) + 1)
    return bce + dice.mean(), (1 - dice.detach()).mean(0)


def prepare_batch(b, device, size, aug=None):
    x = b["x"].to(device, non_blocking=True).float()[:, None] / 255          # B 1 D S S
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(x.shape[2], size, size), mode="trilinear", align_corners=False)
    target = box_masks(b["boxes"].to(device), x.shape[2:])
    if aug is not None:
        x, target = gpu_augment(x, target, b["plane"] == 0, aug)
    return x, target


@torch.no_grad()
def predict(model, loader, ds, device, size):
    model.eval()
    rows, dice = [], []
    for b in loader:
        x, target = prepare_batch(b, device, size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, b["plane"].to(device))
        p = torch.sigmoid(out["cls"].float()).cpu().numpy()
        dice.append(seg_loss(out["seg"][-1], target)[1].cpu().numpy())
        rows += [(i, p[j].tolist()) for j, i in enumerate(b["idx"].tolist())]
    model.train()
    rows = dict(all_gather_lists(rows))
    idx = sorted(rows)
    it = ds.info.iloc[idx]
    p = np.array([rows[i] for i in idx])
    wide = pd.DataFrame(p, columns=ROI_CLASSES).assign(StudyInstanceUID=it.StudyInstanceUID.values,
                                                       SeriesInstanceUID=it.SeriesInstanceUID.values,
                                                       plane=it.Anatomical_Plane.values)
    long = wide.melt(id_vars=["StudyInstanceUID", "SeriesInstanceUID", "plane"], var_name="cls", value_name="p")
    dice = all_reduce_mean({c: float(v) for c, v in zip(ROI_CLASSES, np.mean(dice, 0))}, device)
    return long, wide, dice


def fit(model, train_dl, val_dl, val_ds, studies, args, ctx, sampler):
    device = ctx.device
    log = print if ctx.is_main else (lambda *a, **k: None)
    m = unwrap_model(model)
    unet = [p for n, p in m.named_parameters() if n.startswith(("encoder", "transpconvs", "stages"))]
    ids = {id(p) for p in unet}
    opt = torch.optim.AdamW([{"params": unet, "lr": args.lr_unet},
                             {"params": [p for p in m.parameters() if id(p) not in ids], "lr": args.lr}],
                            weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, total - warm))))
    ema = copy.deepcopy(m).eval() if args.ema else None
    t0, hist, it = time.time(), [], 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        run = {"cls": [], "seg": [], "meta": []}
        for b in train_dl:
            x, target = prepare_batch(b, device, args.size, args.augmentations)
            y = b["y"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, b["plane"].to(device))
            lab = ~torch.isnan(y)
            l_cls = F.binary_cross_entropy_with_logits(out["cls"].float()[lab], y[lab])
            l_seg = sum(seg_loss(s, target)[0] for s in out["seg"]) / len(out["seg"])
            l_meta = F.binary_cross_entropy_with_logits(out["meta"].float(), b["meta"].to(device))
            loss = args.w_cls * l_cls + args.w_seg * l_seg + args.w_meta * l_meta
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()
            sched.step()
            it += 1
            if ema is not None:
                d = min(args.ema, (1 + it) / (10 + it))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), m.parameters()):
                        pe.lerp_(pm, 1 - d)
            run["cls"].append(float(l_cls.detach()))
            run["seg"].append(float(l_seg.detach()))
            run["meta"].append(float(l_meta.detach()))
            if it % args.log_every == 0:
                r = all_reduce_mean({k: float(np.mean(v[-args.log_every:])) for k, v in run.items()}, device)
                log(f"  [{it}/{total}] ep {epoch}  cls {r['cls']:.4f}  seg {r['seg']:.4f}  meta {r['meta']:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:.0f}s")
        long, wide, dice = predict(ema if ema is not None else m, val_dl, val_ds, device, args.size)
        tab, study_tab = study_auc(long, studies)
        log(f"epoch {epoch + 1}  gt {tab['gt'].mean():.4f}  hold {tab['hold'].mean():.4f}  "
            + "  ".join(f"{c[:10]} {r['gt']:.3f}/{r['hold']:.3f}" for c, r in tab.iterrows())
            + f"  | dice {np.mean(list(dice.values())):.3f}  ({time.time() - t0:.0f}s)")
        hist.append(dict(epoch=epoch + 1, gt=tab["gt"].mean(), hold=tab["hold"].mean(),
                         dice=np.mean(list(dice.values()))))
        if ctx.is_main:
            wide.to_csv(os.path.join(args.output_dir, "val_series_preds.csv"), index=False)
            study_tab.to_csv(os.path.join(args.output_dir, "val_study_preds.csv"))
            tab.to_csv(os.path.join(args.output_dir, "val_table.csv"))
            pd.DataFrame(hist).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
    if ctx.is_main:
        torch.save((ema if ema is not None else m).state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
    return tab
