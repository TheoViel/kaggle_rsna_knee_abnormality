"""Fit loop and evaluation.

The metric is the competition's: macro AUC over the 12 classes. Two readings:

  - `gt_auc`:   against the radiologist labels of the 58 GT studies. The real
                metric, but on 58 studies with 9-35 positives per class it moves
                by a couple of points between equivalent checkpoints.
  - `hold_auc`: against the Claude labels, binarised at 0.5, on the report-
                labelled holdout. Steadier, but measures agreement with the
                training labels -- which themselves score ~0.91 on the GT set.

Loss: BCE on the soft Claude labels at the study level, plus `aux_weight` times
the same BCE on each series' own logits (deep supervision: it keeps every
series' attention informative, not just the one the study attention picks).
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from knee_cls.data import gpu_augment, to_device
from params import CLASSES
from training.distributed import all_gather_lists, all_reduce_mean, unwrap_model


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for k, c in enumerate(CLASSES):
        yk = y[:, k]
        ok = ~np.isnan(yk)
        if ok.sum() and 0 < yk[ok].sum() < ok.sum():
            aucs[c] = float(roc_auc_score(yk[ok], p[ok, k]))
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(unwrap_model(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        msd = unwrap_model(model).state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def normalize(x):
    return (x - 0.5) / 0.25


@torch.no_grad()
def predict(model, loader, device, img_size, amp=True):
    """-> DataFrame, one row per study: StudyInstanceUID, pred_*, target_*, gt_*.

    Every rank predicts its shard and the rows are gathered, so every rank must
    call this. DistributedSampler pads with repeats, dropped here.
    """
    model.eval()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        batch["images"] = normalize(gpu_augment(batch, img_size, aug=None))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
        prob = torch.sigmoid(out["logits"].float()).cpu().numpy()
        tgt, gt = batch["target"].cpu().numpy(), batch["gt"].cpu().numpy()
        for i, uid in enumerate(batch["study_uid"]):
            rows.append({"StudyInstanceUID": uid,
                         **{f"pred_{c}": prob[i, k] for k, c in enumerate(CLASSES)},
                         **{f"target_{c}": tgt[i, k] for k, c in enumerate(CLASSES)},
                         **{f"gt_{c}": gt[i, k] for k, c in enumerate(CLASSES)}})
    rows = all_gather_lists(rows)
    return pd.DataFrame(rows).drop_duplicates("StudyInstanceUID").reset_index(drop=True)


def score(df):
    p = df[[f"pred_{c}" for c in CLASSES]].to_numpy()
    gt = df[[f"gt_{c}" for c in CLASSES]].to_numpy()
    tgt = df[[f"target_{c}" for c in CLASSES]].to_numpy()
    is_gt = ~np.isnan(gt).any(1)
    m = {}
    if is_gt.any():
        m["gt_auc"], m["gt_per_class"] = macro_auc(gt[is_gt], p[is_gt])
        # Reference: the training labels themselves, scored the same way.
        m["gt_auc_labels"], _ = macro_auc(gt[is_gt], tgt[is_gt])
    if (~is_gt).any():
        hold = (tgt[~is_gt] >= 0.5).astype(np.float32)
        m["hold_auc"], m["hold_per_class"] = macro_auc(hold, p[~is_gt])
        pt = torch.from_numpy(p[~is_gt]).clamp(1e-6, 1 - 1e-6)
        m["hold_bce"] = float(F.binary_cross_entropy(pt, torch.from_numpy(tgt[~is_gt])))
    return m


def _fmt(m):
    s = []
    for k in ("gt_auc", "hold_auc", "hold_bce"):
        if k in m:
            s.append(f"{k} {m[k]:.4f}")
    return "  ".join(s)


def _fmt_classes(per_class):
    return "  ".join(f"{c[:8]} {v:.3f}" for c, v in per_class.items())


def build_optimizer(model, args):
    m = unwrap_model(model)
    enc = [p for p in m.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc}
    rest = [p for p in m.parameters() if p.requires_grad and id(p) not in enc_ids]
    return torch.optim.AdamW(
        [{"params": enc, "lr": float(args.lr_backbone)},
         {"params": rest, "lr": float(args.lr)}],
        weight_decay=float(args.weight_decay))


def fit(model, train_loader, args, ctx, train_sampler=None, val_loader=None):
    device = ctx.device
    log = print if ctx.is_main else (lambda *a, **k: None)
    amp = bool(args.amp)

    opt = build_optimizer(model, args)
    steps_per_epoch = len(train_loader)
    total = int(args.epochs * steps_per_epoch)
    warmup = max(1, int(float(args.warmup_prop) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warmup if it < warmup
        else 0.5 * (1 + np.cos(np.pi * (it - warmup) / max(1, total - warmup))))
    ema = ModelEMA(model, float(args.ema_decay)) if args.ema_decay else None
    log(f"-> {steps_per_epoch} steps/epoch, {total} steps, warmup {warmup}")

    it, epoch, t0 = 0, 0, time.time()
    running = {"loss": [], "study": [], "series": []}
    history = []
    while it < total:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            if it >= total:
                break
            batch = to_device(batch, device)
            batch["images"] = normalize(gpu_augment(batch, args.img_size, args.augmentations))
            target = batch["target"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(batch)
            l_study = F.binary_cross_entropy_with_logits(out["logits"].float(), target)
            l_series = F.binary_cross_entropy_with_logits(
                out["series_logits"].float(), target[batch["series_study"]])
            loss = l_study + float(args.aux_weight) * l_series

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model, it)
            it += 1
            running["loss"].append(float(loss.detach()))
            running["study"].append(float(l_study.detach()))
            running["series"].append(float(l_series.detach()))

            if it % int(args.log_every) == 0:
                r = all_reduce_mean({k: np.mean(v[-args.log_every:]) for k, v in running.items()},
                                    device)
                log(f"  [{it:>5}/{total}] ep {epoch}  loss {r['loss']:.4f}  "
                    f"study {r['study']:.4f}  series {r['series']:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:.0f}s")
        epoch += 1

        last_epoch = it >= total
        if val_loader is not None and (epoch % int(args.eval_every_epochs) == 0 or last_epoch):
            eval_model = ema.module if ema is not None else model
            df = predict(eval_model, val_loader, device, args.img_size, amp=amp)
            m = score(df)
            m["epoch"] = epoch
            history.append({k: v for k, v in m.items() if not isinstance(v, dict)})
            log(f"  == epoch {epoch}  {_fmt(m)}  ({time.time() - t0:.0f}s)")
            if "gt_per_class" in m:
                log(f"     gt   {_fmt_classes(m['gt_per_class'])}")
            if ctx.is_main:
                df.to_csv(os.path.join(args.output_dir, "val_preds.csv"), index=False)
                pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"),
                                             index=False)

    if ctx.is_main:
        sd = (ema.module if ema is not None else unwrap_model(model)).state_dict()
        torch.save(sd, os.path.join(args.output_dir, "ckpt_final.pt"))
    if val_loader is not None and history:
        log(f"  -> labels-as-predictions reference on GT: {m.get('gt_auc_labels', float('nan')):.4f}")
        if "hold_per_class" in m:
            log(f"     hold {_fmt_classes(m['hold_per_class'])}")
    return history[-1] if history else {}
