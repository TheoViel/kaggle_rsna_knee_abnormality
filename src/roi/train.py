"""3D U-Net for the ROI boxes: model, partial-label loss, fit loop, box predictions.

The network predicts, per class, the probability that a voxel is inside that
class's box. Every series has exactly one ROI per class, so exactly one box is read back per
class, always: the bounding box of the most probable connected component above
`threshold`, or, when no voxel reaches it, above half the class's peak
probability. Its confidence is the peak probability; a box below `threshold`
is a model failure (counted as such), not a missing ROI (`prob_to_boxes`).

Validation reports three things, per class:
  - against the labels (where the attention gave one): detection rate and the
    3D IoU of predicted vs label box, in the volume grid;
  - failures: share of (series, class) whose box is below the confidence
    threshold. Every series has every ROI, so this should be ~0;
  - cross-view agreement of the predicted boxes, in patient space, between the
    sagittal and coronal series of a study (`roi.boxes.cross_view`): the check
    that needs no labels at all.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

from roi.boxes import cross_view
from roi.data import flip_axis, gpu_augment
from roi.extract import ROI_CLASSES
from roi.geometry import from_volume
from training.distributed import all_gather_lists, all_reduce_mean, unwrap_model


def build_model(num_classes=len(ROI_CLASSES), features=(32, 64, 128, 256, 320, 320)):
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    # (D, S, S) = (32, 128, 128) with few, thick slices: pool in-plane first, and
    # 1x3x3 kernels until the in-plane resolution is closer to the slice spacing.
    return ResidualEncoderUNet(
        input_channels=1, n_stages=6, features_per_stage=list(features), conv_op=nn.Conv3d,
        kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6], num_classes=num_classes,
        n_conv_per_stage_decoder=[1] * 5, conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True})


def masked_loss(logits, target, labelled):
    """BCE + soft Dice over the labelled (sample, class) pairs only."""
    m = labelled.float()
    if m.sum() == 0:
        return logits.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((2, 3, 4))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3, 4))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3, 4)) + target.sum((2, 3, 4)) + 1)
    return ((bce + dice) * m).sum() / m.sum()


def prob_to_boxes(prob, threshold):
    """(K, D, S, S) probabilities -> per class (box in volume voxels, peak prob, mean prob).

    Always one box per class (see module doc). The mean probability over the
    box's component is the sharper confidence: the peak is ~1 almost everywhere.
    """
    out = []
    for p in prob:
        conf = float(p.max())
        region = p >= (threshold if conf >= threshold else 0.5 * conf)
        lab, n = ndi.label(region, structure=np.ones((3, 3, 3)))
        mass = ndi.sum(p, lab, index=np.arange(1, n + 1))
        comp = lab == int(np.argmax(mass)) + 1
        zs, ys, xs = np.nonzero(comp)
        out.append((dict(z0=zs.min(), z1=zs.max(), y0=ys.min(), y1=ys.max(),
                         x0=xs.min(), x1=xs.max()), conf, float(p[comp].mean())))
    return out


def _vol_iou(a, b):
    lo = [max(a[f"{k}0"], b[f"{k}0"]) for k in "zyx"]
    hi = [min(a[f"{k}1"], b[f"{k}1"]) for k in "zyx"]
    inter = np.prod([max(0, h - l + 1) for l, h in zip(lo, hi)])
    va = np.prod([a[f"{k}1"] - a[f"{k}0"] + 1 for k in "zyx"])
    vb = np.prod([b[f"{k}1"] - b[f"{k}0"] + 1 for k in "zyx"])
    return inter / (va + vb - inter)


@torch.no_grad()
def predict(model, loader, index, device, threshold=0.5, with_labels=True):
    """Boxes for every series in `loader`, gathered over ranks.

    Returns a DataFrame: one row per (series, class) with conf, the predicted box
    in original voxels and, when the series had a label for the class, the IoU
    with it.
    """
    model.eval()
    info = index.set_index("row")
    rows = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True).float() / 255
        with torch.autocast(device.type, dtype=torch.bfloat16):
            prob = torch.sigmoid(model(image).float()).cpu().numpy()
        target = batch["target"].numpy() if with_labels else None
        # Back from the canonical (right-knee) orientation to the series' own.
        for i in torch.nonzero(batch["flipped"]).flatten().tolist():
            ax = flip_axis(bool(batch["sagittal"][i])) + 1          # + class axis
            prob[i] = np.flip(prob[i], ax)
            if target is not None:
                target[i] = np.flip(target[i], ax)
        for i, row in enumerate(batch["row"].tolist()):
            r = info.loc[row]
            for k, (vb, conf, p_mean) in enumerate(prob_to_boxes(prob[i], threshold)):
                rec = dict(SeriesInstanceUID=r.SeriesInstanceUID, StudyInstanceUID=r.StudyInstanceUID,
                           plane=r.Anatomical_Plane, **{"class": ROI_CLASSES[k]}, conf=conf,
                           p_mean=p_mean)
                rec.update(from_volume(vb, r.n_slices, r.n_cache, r.rows, r.cols, *prob.shape[2:4]))
                if with_labels and bool(batch["labelled"][i, k]):
                    zs, ys, xs = np.nonzero(target[i, k])
                    lb = dict(z0=zs.min(), z1=zs.max(), y0=ys.min(), y1=ys.max(), x0=xs.min(), x1=xs.max())
                    rec["label_iou"] = _vol_iou(vb, lb)
                rows.append(rec)
    model.train()
    return pd.DataFrame(all_gather_lists(rows)).drop_duplicates(["SeriesInstanceUID", "class"])


def score(pred, min_conf=0.5):
    """Per-class validation table (see module doc)."""
    has = pred.conf >= min_conf
    boxes = pred[has & pred.z0.notna()].copy()
    for c in ["z0", "z1", "y0", "y1", "x0", "x1"]:
        boxes[c] = boxes[c].astype(int)
    cv = cross_view(boxes)
    t = {}
    for c in ROI_CLASSES:
        p = pred[pred["class"] == c]
        lab = p[p.get("label_iou", pd.Series(dtype=float)).notna()] if "label_iou" in p else p.iloc[:0]
        pairs = cv[cv.cls == c]
        t[c] = dict(label_n=len(lab), label_iou=lab.label_iou.mean() if len(lab) else np.nan,
                    failures=(p.conf < min_conf).mean(),
                    xview_overlap=pairs.overlap.mean() if len(pairs) else np.nan,
                    xview_iou=pairs.iou.mean() if len(pairs) else np.nan)
    t = pd.DataFrame(t).T
    t.loc["ALL"] = t.mean(numeric_only=True)
    return t


def fit(model, train_loader, val_loader, index, args, ctx, sampler=None):
    device = ctx.device
    log = print if ctx.is_main else (lambda *a, **k: None)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total = int(args.epochs * len(train_loader))
    warmup = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warmup if it < warmup
        else 0.5 * (1 + np.cos(np.pi * (it - warmup) / max(1, total - warmup))))
    log(f"-> {len(train_loader)} steps/epoch, {total} steps")

    it, epoch, t0, running = 0, 0, time.time(), []
    table = None
    while it < total:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_loader:
            if it >= total:
                break
            image = batch["image"].to(device, non_blocking=True).float() / 255
            target = batch["target"].to(device, non_blocking=True).float()
            labelled = batch["labelled"].to(device, non_blocking=True)
            image, target = gpu_augment(image, target, batch["sagittal"], args.augmentations)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(image)
            loss = masked_loss(logits.float(), target, labelled)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()
            sched.step()
            running.append(float(loss.detach()))
            it += 1
            if it % int(args.log_every) == 0:
                r = all_reduce_mean({"loss": float(np.mean(running[-args.log_every:]))}, device)
                log(f"  [{it:>6}/{total}] ep {epoch}  loss {r['loss']:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:.0f}s")
        epoch += 1
        if epoch % int(args.eval_every_epochs) == 0 or it >= total:
            pred = predict(unwrap_model(model), val_loader, index, device, args.threshold)
            table = score(pred)
            log(f"  == epoch {epoch}  ({time.time() - t0:.0f}s)\n{table.round(3).to_string()}")
            if ctx.is_main:
                pred.to_csv(os.path.join(args.output_dir, "val_pred.csv"), index=False)
    if ctx.is_main:
        torch.save(unwrap_model(model).state_dict(), os.path.join(args.output_dir, "ckpt_final.pt"))
    return table
